#!/usr/bin/env python3
"""
Step 0: build the train / test split manifest for cpg0039-garcia-fossa-livecellpainting.

Huh-7 cells stained with Acridine Orange, imaged in two channels: AO green (GFP
filter) and AO red (Propidium Iodide filter).

  Test, DILI       every imaged well of the three deposited DILI plates (Figure 4),
                   fields 1, 9, 17 and 25 of 25.
  Test, AgNP       the Ag100 and Non-treated wells of the three AgNP plates listed
                   in test_wells_agnp.csv (Figure 3), fields 1, 6, 11 and 16 of 16.
  Train            n_train FOVs spread evenly over the remaining AgNP wells and
                   the AO-stained wells of the Acidification_AO plates, distinct
                   random fields per well.

The DILI experiment is never used for training. Image paths are found by
parsing the Cytation 5 file names, ``{well}_{n}_{channel_index}_{field}_{filter}_001.tif``.

Outputs:
  {out_dir}/Huh7/split_manifest.csv   one row per (plate, well, field, role)
  {out_dir}/Huh7/split_summary.json   counts and overlap checks

Usage:
    python 0_split.py
    python 0_split.py --n_train 300 --dili_fields 1 9 17 25
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

TRAIN_ROLE = "train"
CELL_LINE = "Huh7"
FILTERS = {"AO_Green": "GFP", "AO_Red": "Propidium Iodide"}
PATH_COLUMNS = {ch: f"path_{ch}" for ch in FILTERS}
FILE_PATTERN = re.compile(r"^(?P<well>[A-H]\d{1,2})_\d+_\d+_(?P<site>\d+)_(?P<filter>.+)_001\.tif$")

DILI_BATCH = "2022_06_07_DILI"
AGNP_BATCH = "2021_10_08_AgNPViability"
ACIDIFICATION_BATCH = "2022_03_24_Acidification_AO"
AGNP_TEST_FIELDS = (1, 6, 11, 16)

DEFAULT_DATA_DIR = "/project/cell_paint_mono/cpg0039-garcia-fossa-livecellpainting/unicamp"
DEFAULT_OUT_DIR = "/project/cell_paint_mono/splits/cpg0039-garcia-fossa-livecellpainting"

MANIFEST_COLUMNS = [
    "role", "plate_id", "cell_line", "test_set", "batch", "well", "site",
    "compound", "np_size_nm", "concentration", "control_type", *PATH_COLUMNS.values(),
]


def index_plate_images(plate_dir: Path) -> pd.DataFrame:
    """FOVs of one plate that have both AO channels, with their image paths.

    Parameters
    ----------
    plate_dir : Path
        Directory holding the plate's TIFF files.

    Returns
    -------
    pandas.DataFrame
        One row per (well, site) with one ``path_{channel}`` column per channel.
    """
    records = []
    for name in sorted(p.name for p in plate_dir.iterdir()):
        match = FILE_PATTERN.match(name)
        if match and match["filter"] in FILTERS.values():
            records.append({
                "well": match["well"], "site": int(match["site"]),
                "filter": match["filter"], "path": str(plate_dir / name),
            })
    images = pd.DataFrame(records).pivot(index=["well", "site"], columns="filter", values="path")
    images = images.dropna().rename(columns={f: PATH_COLUMNS[ch] for ch, f in FILTERS.items()})
    return images.reset_index()


def load_platemap(data_dir: Path, batch: str) -> pd.DataFrame:
    """Platemap of every plate of a batch, one row per (plate_id, well)."""
    platemaps = data_dir / "workspace" / "metadata" / "platemaps" / batch
    barcodes = pd.read_csv(platemaps / "barcode_platemap.csv")
    frames = []
    for barcode, name in zip(barcodes.Assay_Plate_Barcode, barcodes.Plate_Map_Name):
        pm = pd.read_csv(platemaps / "platemap" / f"{name}.txt", sep="\t")
        frames.append(pm.assign(plate_id=barcode))
    pm = pd.concat(frames, ignore_index=True).rename(columns={"well_position": "well"})
    concentration = pm.Concentration_uM if "Concentration_uM" in pm else pm.Concentration
    return pd.DataFrame({
        "plate_id": pm.plate_id,
        "well": pm.well,
        "compound": pm.Compound,
        "np_size_nm": pm.NPSize_nm if "NPSize_nm" in pm else np.nan,
        "concentration": concentration,
        "control_type": pm.control_type.fillna(""),
        "dye": pm.Dye if "Dye" in pm else "AO",
    })


def load_batch_fovs(data_dir: Path, batch: str) -> pd.DataFrame:
    """All FOVs with both AO channels of one batch, joined to the platemap."""
    images_dir = data_dir / "images" / batch / "images"
    fovs = pd.concat(
        [index_plate_images(d).assign(plate_id=d.name) for d in sorted(images_dir.iterdir()) if d.is_dir()],
        ignore_index=True,
    )
    fovs = fovs.merge(load_platemap(data_dir, batch), on=["plate_id", "well"], how="left")
    return fovs.assign(batch=batch, cell_line=CELL_LINE)


def sample_training_fovs(pool: pd.DataFrame, n_train: int, rng: np.random.Generator) -> pd.DataFrame:
    """``n_train`` FOVs spread evenly over the wells of ``pool``, distinct random fields per well."""
    wells = pool.groupby(["plate_id", "well"])
    keys = sorted(wells.groups)
    per_well = np.full(len(keys), n_train // len(keys))
    per_well[rng.choice(len(keys), size=n_train % len(keys), replace=False)] += 1
    picks = [
        wells.get_group(key).sample(n=n, random_state=rng)
        for key, n in zip(keys, per_well) if n > 0
    ]
    return pd.concat(picks)


def assign_test_roles(rows: pd.DataFrame, test_set: str) -> pd.DataFrame:
    """Label test rows as perturbation or negative control."""
    role = np.where(rows.control_type == "negcon", "test_negcon", "test_perturbation")
    return rows.assign(role=role, test_set=test_set)


def build_manifest(data_dir: Path, agnp_test_wells: pd.DataFrame, dili_fields: list[int],
                   n_train: int, rng: np.random.Generator) -> pd.DataFrame:
    """Test rows of both test sets and ``n_train`` random training rows."""
    dili = load_batch_fovs(data_dir, DILI_BATCH)
    dili_test = assign_test_roles(dili[dili.site.isin(dili_fields)], "DILI")

    agnp = load_batch_fovs(data_dir, AGNP_BATCH)
    test_keys = set(zip(agnp_test_wells.Metadata_Plate, agnp_test_wells.Metadata_Well))
    is_test_well = np.array([k in test_keys for k in zip(agnp.plate_id, agnp.well)])
    agnp_test = agnp[is_test_well & agnp.site.isin(AGNP_TEST_FIELDS)]
    n_found = agnp_test.groupby(["plate_id", "well"]).ngroups
    assert n_found == len(test_keys), f"found {n_found} of {len(test_keys)} AgNP test wells"
    agnp_test = assign_test_roles(agnp_test, "AgNP")

    acidification = load_batch_fovs(data_dir, ACIDIFICATION_BATCH)
    pool = pd.concat([agnp[~is_test_well], acidification[acidification.dye == "AO"]], ignore_index=True)
    train = sample_training_fovs(pool, n_train, rng).assign(role=TRAIN_ROLE, test_set="")

    manifest = pd.concat([dili_test, agnp_test, train], ignore_index=True)
    return manifest[MANIFEST_COLUMNS].sort_values(["role", "plate_id", "well", "site"])


def check_manifest(manifest: pd.DataFrame) -> dict:
    """Assert that no training well is a test well and DILI is never trained on."""
    train = manifest[manifest.role == TRAIN_ROLE]
    test = manifest[manifest.role != TRAIN_ROLE]
    overlap = set(zip(train.plate_id, train.well)) & set(zip(test.plate_id, test.well))
    assert not overlap, f"train/test well overlap: {sorted(overlap)[:5]}"
    assert not (train.batch == DILI_BATCH).any()
    return {"well_overlap": 0}


def summarize(manifest: pd.DataFrame, checks: dict, args: argparse.Namespace) -> dict:
    """Counts behind the split, written next to the manifest."""
    def counts(df: pd.DataFrame) -> dict:
        return {
            "fovs": len(df),
            "wells": df.groupby(["plate_id", "well"]).ngroups,
            "by_batch": df.groupby("batch").size().to_dict(),
        }

    return {
        "cell_line": CELL_LINE,
        "args": {"n_train": args.n_train, "seed": args.seed, "dili_fields": args.dili_fields,
                 "agnp_fields": list(AGNP_TEST_FIELDS)},
        "train": counts(manifest[manifest.role == TRAIN_ROLE]),
        "test_perturbation": counts(manifest[manifest.role == "test_perturbation"]),
        "test_negcon": counts(manifest[manifest.role == "test_negcon"]),
        "checks": checks,
    }


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Build the cpg0039 train / test split manifest")
    ap.add_argument("--n_train", type=int, default=300, help="training FOVs")
    ap.add_argument("--dili_fields", type=int, nargs="+", default=[1, 9, 17, 25])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--agnp_test_wells", default=str(Path(__file__).parent / "test_wells_agnp.csv"))
    ap.add_argument("--data_dir", default=DEFAULT_DATA_DIR)
    ap.add_argument("--out_dir", default=DEFAULT_OUT_DIR)
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    manifest = build_manifest(
        Path(args.data_dir), pd.read_csv(args.agnp_test_wells), args.dili_fields,
        args.n_train, np.random.default_rng(args.seed),
    )
    checks = check_manifest(manifest)

    out_dir = Path(args.out_dir) / CELL_LINE
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(out_dir / "split_manifest.csv", index=False)
    summary = summarize(manifest, checks, args)
    with open(out_dir / "split_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)

    print(f"{CELL_LINE} -> {out_dir}")
    for role in ("test_perturbation", "test_negcon", "train"):
        c = summary[role]
        print(f"  {role:18s} {c['fovs']:5d} FOVs on {c['wells']:4d} wells {c['by_batch']}")


if __name__ == "__main__":
    main()
