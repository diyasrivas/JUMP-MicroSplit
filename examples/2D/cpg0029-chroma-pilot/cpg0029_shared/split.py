"""
Test set selection and per-dye-set split manifests for cpg0029.

All five plates share the platemap ``S-C-7314-04-CMP-001`` (90 compounds x 4
wells + 24 DMSO wells), so one set of test FOVs (well, site) is drawn once and
reused on every plate. Training FOVs are drawn per plate from the wells that are
not in the test set.
"""

import re
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import tifffile

from .dye_sets import CELL_LINE, DATA_DIR, DyeSet

PLATEMAP = (DATA_DIR / "workspace" / "metadata" / "platemaps" / "2023_05_15_Batch1"
            / "platemap" / "S-C-7314-04-CMP-001.txt")
SITES_PER_WELL = 9
TRAIN_ROLE = "train"
FILE_PATTERN = re.compile(r"^r(?P<row>\d{2})c(?P<col>\d{2})f(?P<site>\d{2})p01-ch(?P<ch>\d+)sk1fk1fl1\.tiff$")
TEST_COLUMNS = ["well", "site", "broad_sample", "compound", "moa", "control_type"]


def load_platemap(path: Path = PLATEMAP) -> pd.DataFrame:
    """One row per well: ``well, broad_sample, compound, moa, control_type``."""
    pm = pd.read_csv(path, sep="\t")
    return pd.DataFrame({
        "well": pm.well_position,
        "broad_sample": pm.broad_sample.fillna(""),
        "compound": pm["Common Name"].fillna(""),
        "moa": pm.MoA.fillna(""),
        "control_type": pm.control_type.fillna(""),
    })


def select_test_fovs(n_compounds: int, n_dmso_wells: int, n_sites: int, seed: int) -> pd.DataFrame:
    """Random compounds (all 4 wells each), random DMSO wells and random sites per well.

    Compounds are drawn uniformly from the 90 treatments, without stratifying by
    MoA or by the paper's results. Each test well gets ``n_sites`` distinct
    random sites out of 9.
    """
    pm = load_platemap()
    rng = np.random.default_rng(seed)
    treatments = sorted(pm.loc[pm.control_type == "", "broad_sample"].unique())
    compounds = rng.choice(treatments, size=n_compounds, replace=False)
    dmso_wells = rng.choice(sorted(pm.loc[pm.control_type == "negcon", "well"]),
                            size=n_dmso_wells, replace=False)
    wells = pm[pm.broad_sample.isin(compounds) & (pm.control_type == "")
               | pm.well.isin(dmso_wells)].sort_values("well")

    rows = []
    for record in wells.to_dict("records"):
        sites = np.sort(rng.choice(np.arange(1, SITES_PER_WELL + 1), size=n_sites, replace=False))
        rows += [{**record, "site": int(s)} for s in sites]
    return pd.DataFrame(rows)[TEST_COLUMNS]


def well_name(row: int, col: int) -> str:
    return f"{chr(ord('A') + row - 1)}{col:02d}"


def index_plate(dye_set: DyeSet) -> pd.DataFrame:
    """FOVs of the plate that have every channel the dye set needs.

    Returns one row per (well, site) with ``image_prefix`` and one
    ``path_{channel}`` column per unmixed and raw-only channel.
    """
    needed = {**dye_set.unmixed, **dye_set.raw_only}
    by_index = {idx: name for name, idx in needed.items()}
    records = []
    for name in sorted(p.name for p in dye_set.images_dir.iterdir()):
        m = FILE_PATTERN.match(name)
        if m and int(m["ch"]) in by_index:
            records.append({
                "well": well_name(int(m["row"]), int(m["col"])),
                "site": int(m["site"]),
                "image_prefix": name.split("-ch")[0],
                "channel": by_index[int(m["ch"])],
                "path": str(dye_set.images_dir / name),
            })
    long = pd.DataFrame(records)
    wide = long.pivot(index=["well", "site", "image_prefix"], columns="channel", values="path")
    wide = wide.dropna(subset=list(needed)).rename(columns={ch: f"path_{ch}" for ch in needed})
    return wide.reset_index()[["well", "site", "image_prefix", *[f"path_{ch}" for ch in needed]]]


def unreadable_fovs(fovs: pd.DataFrame) -> List[tuple]:
    """(well, site) of FOVs with at least one TIFF that cannot be opened."""
    bad = []
    path_cols = [c for c in fovs.columns if c.startswith("path_")]
    for r in fovs.itertuples(index=False):
        for col in path_cols:
            try:
                with tifffile.TiffFile(getattr(r, col)) as tif:
                    tif.pages[0].shape
            except Exception:
                bad.append((r.well, r.site))
                break
    return bad


def sample_training_fovs(pool: pd.DataFrame, n_train: int, rng: np.random.Generator) -> pd.DataFrame:
    """``n_train`` FOVs spread evenly over the wells of ``pool``, distinct random sites per well."""
    wells = pool.groupby("well")
    keys = sorted(wells.groups)
    per_well = np.full(len(keys), n_train // len(keys))
    per_well[rng.choice(len(keys), size=n_train % len(keys), replace=False)] += 1
    picks = [wells.get_group(k).sample(n=min(n, len(wells.get_group(k))), random_state=rng)
             for k, n in zip(keys, per_well) if n > 0]
    return pd.concat(picks)


def build_manifest(dye_set: DyeSet, test_fovs: pd.DataFrame, n_train: int, seed: int,
                   verify: bool = True) -> tuple[pd.DataFrame, Dict]:
    """Split manifest of one dye set and a dict of checks for the summary.

    Test FOVs missing or unreadable on this plate are dropped and reported.
    Training FOVs come from the non-test wells of the same plate; an unreadable
    training FOV is replaced by another random FOV from the pool. Only the
    selected FOVs are opened, not the whole plate.
    """
    pm = load_platemap()
    fovs = index_plate(dye_set)
    rng = np.random.default_rng(seed)
    bad = set()

    keys = test_fovs[["well", "site"]]
    test = keys.merge(fovs, on=["well", "site"], how="inner")
    if verify:
        bad |= set(unreadable_fovs(test))
        test = test[[(w, s) not in bad for w, s in zip(test.well, test.site)]]
    test = test.merge(pm, on="well", how="left")
    missing = sorted(set(zip(keys.well, keys.site)) - set(zip(test.well, test.site)))
    test["role"] = np.where(test.control_type == "negcon", "test_negcon", "test_perturbation")

    pool = fovs[~fovs.well.isin(set(test_fovs.well))]
    train = sample_training_fovs(pool, n_train, rng)
    while verify:
        new_bad = set(unreadable_fovs(train))
        if not new_bad:
            break
        bad |= new_bad
        train = train[[(w, s) not in bad for w, s in zip(train.well, train.site)]]
        taken = set(zip(train.well, train.site)) | bad
        rest = pool[[(w, s) not in taken for w, s in zip(pool.well, pool.site)]]
        train = pd.concat([train, rest.sample(n=n_train - len(train), random_state=rng)])
    train = train.merge(pm, on="well", how="left").assign(role=TRAIN_ROLE)

    manifest = pd.concat([test, train], ignore_index=True).assign(
        plate_id=dye_set.barcode, dye_set=dye_set.name, batch=dye_set.batch, cell_line=CELL_LINE,
    )
    path_cols = [c for c in fovs.columns if c.startswith("path_")]
    columns = ["role", "plate_id", "dye_set", "cell_line", "batch", "well", "site", "image_prefix",
               "broad_sample", "compound", "moa", "control_type", *path_cols]
    manifest = manifest[columns].sort_values(["role", "well", "site"]).reset_index(drop=True)

    overlap = set(manifest.loc[manifest.role == TRAIN_ROLE, "well"]) & set(test_fovs.well)
    assert not overlap, f"train/test well overlap: {sorted(overlap)[:5]}"
    checks = {
        "well_overlap": 0,
        "unreadable_fovs_dropped": [list(k) for k in sorted(bad)],
        "test_fovs_missing_on_plate": [list(k) for k in missing],
    }
    return manifest, checks


def summarize(manifest: pd.DataFrame, checks: Dict, dye_set: DyeSet, args: Dict) -> Dict:
    def counts(df: pd.DataFrame) -> Dict:
        return {"fovs": len(df), "wells": df.well.nunique(), "compounds": df.broad_sample.nunique()}

    return {
        "dye_set": dye_set.name,
        "label": dye_set.label,
        "barcode": dye_set.barcode,
        "unmixed_channels": dye_set.unmixed,
        "raw_only_channels": dye_set.raw_only,
        "args": args,
        "train": counts(manifest[manifest.role == TRAIN_ROLE]),
        "test_perturbation": counts(manifest[manifest.role == "test_perturbation"]),
        "test_negcon": counts(manifest[manifest.role == "test_negcon"]),
        "checks": checks,
    }
