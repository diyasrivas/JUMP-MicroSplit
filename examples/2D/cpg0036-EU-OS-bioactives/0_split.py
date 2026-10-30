#!/usr/bin/env python3
"""
Step 0: build the train / test split manifest for cpg0036-EU-OS-bioactives.

Test: the wells listed in ``test_wells.csv`` (library plate B1001, replicates
R1 to R4, at every partner site and for both cell lines: 40 compounds, all
Tetrandrine and Nocodazole wells and 12 DMSO wells), fields 2, 4, 6 and 8.

Train: ``n_train`` FOVs per cell line from library plates B1002 to B1007, spread
evenly over the partner sites that imaged the cell line and over their plates;
distinct wells, one random field each. Positive-control wells are excluded and
library plate B1001 is never used for training.

Image paths are taken from the authors' CellProfiler ``load_data.csv`` files,
which record the channel order of every site (it differs between sites and
between the two FMP cell lines).

Outputs (per cell line):
  {out_dir}/{cell_line}/split_manifest.csv   one row per (plate, well, field, role)
  {out_dir}/{cell_line}/split_summary.json   counts and overlap checks

Usage:
    python 0_split.py
    python 0_split.py --cell_lines HepG2 --n_train 300
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

TRAIN_ROLE = "train"
PARTNERS = ("FMP", "IMTM", "MEDINA", "USC")
CELL_LINES = ("HepG2", "U2OS")
LOAD_DATA_COLUMNS = {
    "DNA": "URL_OrigDNA",
    "ER_RNA": "URL_OrigER",
    "Actin_AGP": "URL_OrigAGP",
    "Mito": "URL_OrigMito",
}
GALLERY_PREFIX = "s3://cellpainting-gallery/cpg0036-EU-OS-bioactives"
TEST_LIBRARY_PLATE = "B1001"
NEGATIVE_CONTROL = "DMSO"
POSITIVE_CONTROLS = ("Tetrandrine", "Nocodazole")
TEST_FIELDS = (2, 4, 6, 8)

PATH_COLUMNS = {ch: f"path_{ch}" for ch in LOAD_DATA_COLUMNS}

DEFAULT_DATA_DIR = "/project/cell_paint_mono/cpg0036-EU-OS-bioactives"
DEFAULT_OUT_DIR = "/project/cell_paint_mono/splits/cpg0036-EU-OS-bioactives"

MANIFEST_COLUMNS = [
    "role", "plate_id", "cell_line", "partner", "batch", "plate_map_name",
    "well", "site", "eos", "concentration",
    *PATH_COLUMNS.values(),
]


def load_partner_fovs(data_dir: Path, partner: str) -> pd.DataFrame:
    """All FOVs of one partner site with their per-channel image paths and platemap fields.

    Parameters
    ----------
    data_dir : Path
        Root of the local copy of the dataset.
    partner : str
        Partner site, e.g. ``"FMP"``.

    Returns
    -------
    pandas.DataFrame
        One row per FOV.
    """
    workspace = data_dir / partner / "workspace"
    frames = []
    for path in sorted((workspace / "load_data_csv").glob("*/*/load_data.csv")):
        batch = path.parent.parent.name
        load_data = pd.read_csv(path)
        fovs = pd.DataFrame({
            "cell_line": batch.rsplit("_", 1)[-1],
            "partner": partner,
            "batch": batch,
            "plate_map_name": load_data.Metadata_Plate,
            "well": load_data.Metadata_Well,
            "site": load_data.Metadata_Site.astype(int),
        })
        for channel, column in LOAD_DATA_COLUMNS.items():
            fovs[PATH_COLUMNS[channel]] = load_data[column].str.replace(
                GALLERY_PREFIX, str(data_dir), regex=False
            )
        frames.append(fovs)
    fovs = pd.concat(frames, ignore_index=True)

    platemaps = pd.concat(
        [
            pd.read_csv(p, sep=r"\s+").assign(batch=p.parent.parent.name)
            for p in sorted((workspace / "metadata" / "platemaps").glob("*/platemap/*.txt"))
        ],
        ignore_index=True,
    ).rename(columns={
        "well_position": "well", "Metadata_EOS": "eos", "Metadata_Concentration": "concentration",
    })
    fovs = fovs.merge(
        platemaps[["batch", "plate_map_name", "well", "eos", "concentration"]],
        on=["batch", "plate_map_name", "well"], how="left",
    )
    fovs["plate_id"] = fovs.partner + "_" + fovs.cell_line + "_" + fovs.plate_map_name
    fovs["library_plate"] = fovs.plate_map_name.str.split("_").str[0]
    return fovs


def test_rows(fovs: pd.DataFrame, test_wells: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """FOVs of the test wells at the test fields.

    Returns the rows and the number of test FOVs absent from the deposited data.
    Every test well must have at least one of its test fields.
    """
    wells = test_wells.rename(columns={
        "Metadata_Partner": "partner", "Metadata_CellLine": "cell_line", "Metadata_Well": "well",
    })
    wells["plate_map_name"] = wells.Metadata_Plate + "_" + wells.Metadata_Replicate
    keys = ["partner", "cell_line", "plate_map_name", "well"]
    rows = fovs[fovs.site.isin(TEST_FIELDS)].merge(wells[keys], on=keys)
    n_wells_found = rows.groupby(keys).ngroups
    assert n_wells_found == len(wells), f"found {n_wells_found} of {len(wells)} test wells"
    rows["role"] = np.where(rows.eos == NEGATIVE_CONTROL, "test_negcon", "test_perturbation")
    return rows, len(wells) * len(TEST_FIELDS) - len(rows)


def allocate(total: int, n_groups: int, rng: np.random.Generator) -> np.ndarray:
    """Split ``total`` into ``n_groups`` near-equal integers, remainder to random groups."""
    counts = np.full(n_groups, total // n_groups)
    counts[rng.choice(n_groups, size=total % n_groups, replace=False)] += 1
    return counts


def train_rows(fovs: pd.DataFrame, n_train: int, rng: np.random.Generator) -> pd.DataFrame:
    """``n_train`` training FOVs, even over partners, then plates; distinct wells, one field each."""
    pool = fovs[
        (fovs.library_plate != TEST_LIBRARY_PLATE)
        & fovs.eos.notna()
        & ~fovs.eos.isin(POSITIVE_CONTROLS)
    ]
    partners = sorted(pool.partner.unique())
    picks = []
    for partner, n_partner in zip(partners, allocate(n_train, len(partners), rng)):
        plates = pool[pool.partner == partner].groupby("plate_id")
        plate_ids = sorted(plates.groups)
        for plate_id, n_plate in zip(plate_ids, allocate(n_partner, len(plate_ids), rng)):
            plate = plates.get_group(plate_id)
            wells = rng.choice(np.sort(plate.well.unique()), size=n_plate, replace=False)
            for well in wells:
                candidates = plate[plate.well == well]
                picks.append(candidates.iloc[rng.integers(len(candidates))])
    rows = pd.DataFrame(picks)
    rows["role"] = TRAIN_ROLE
    return rows


def check_manifest(manifest: pd.DataFrame) -> dict:
    """Assert that the test plates are never used for training; return check counts."""
    train = manifest[manifest.role == TRAIN_ROLE]
    test = manifest[manifest.role != TRAIN_ROLE]
    overlap = set(zip(train.plate_id, train.well)) & set(zip(test.plate_id, test.well))
    assert not overlap, f"train/test well overlap: {sorted(overlap)[:5]}"
    assert not train.plate_map_name.str.startswith(TEST_LIBRARY_PLATE).any()
    assert not train.eos.isin(POSITIVE_CONTROLS).any()
    paths = manifest[list(PATH_COLUMNS.values())].values.ravel()
    missing = [p for p in paths if not Path(p).exists()]
    return {"well_overlap": 0, "missing_images": len(missing), "missing_examples": missing[:3]}


def summarize(manifest: pd.DataFrame, checks: dict, args: argparse.Namespace) -> dict:
    """Counts behind a split, written next to the manifest."""
    def counts(df: pd.DataFrame) -> dict:
        return {
            "fovs": len(df),
            "wells": df.groupby(["plate_id", "well"]).ngroups,
            "by_partner": df.groupby("partner").size().to_dict(),
        }

    return {
        "cell_line": manifest.cell_line.iloc[0],
        "args": {"n_train": args.n_train, "seed": args.seed, "test_fields": list(TEST_FIELDS)},
        "train": counts(manifest[manifest.role == TRAIN_ROLE]),
        "test_perturbation": counts(manifest[manifest.role == "test_perturbation"]),
        "test_negcon": counts(manifest[manifest.role == "test_negcon"]),
        "checks": checks,
    }


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Build the cpg0036 train / test split manifest")
    ap.add_argument("--cell_lines", nargs="+", default=list(CELL_LINES), choices=CELL_LINES)
    ap.add_argument("--n_train", type=int, default=300, help="training FOVs per cell line")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--test_wells", default=str(Path(__file__).parent / "test_wells.csv"))
    ap.add_argument("--data_dir", default=DEFAULT_DATA_DIR)
    ap.add_argument("--out_dir", default=DEFAULT_OUT_DIR)
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    data_dir = Path(args.data_dir)
    fovs = pd.concat([load_partner_fovs(data_dir, p) for p in PARTNERS], ignore_index=True)
    test_wells = pd.read_csv(args.test_wells)

    for cell_line in args.cell_lines:
        rng = np.random.default_rng([args.seed, CELL_LINES.index(cell_line)])
        line_fovs = fovs[fovs.cell_line == cell_line]
        line_wells = test_wells[test_wells.Metadata_CellLine == cell_line]
        test, n_test_not_deposited = test_rows(line_fovs, line_wells)
        manifest = pd.concat(
            [test, train_rows(line_fovs, args.n_train, rng)], ignore_index=True,
        )[MANIFEST_COLUMNS].sort_values(["role", "plate_id", "well", "site"])
        checks = check_manifest(manifest)
        checks["test_fovs_not_deposited"] = n_test_not_deposited
        if n_test_not_deposited:
            print(f"NOTE: {n_test_not_deposited} test FOVs are not in the deposited data")
        if checks["missing_images"]:
            print(f"WARNING: {checks['missing_images']} image files not found, "
                  f"e.g. {checks['missing_examples']}")

        out_dir = Path(args.out_dir) / cell_line
        out_dir.mkdir(parents=True, exist_ok=True)
        manifest.to_csv(out_dir / "split_manifest.csv", index=False)
        summary = summarize(manifest, checks, args)
        with open(out_dir / "split_summary.json", "w") as f:
            json.dump(summary, f, indent=2, default=str)

        print(f"{cell_line} -> {out_dir}")
        for role in ("test_perturbation", "test_negcon", "train"):
            c = summary[role]
            print(f"  {role:18s} {c['fovs']:5d} FOVs on {c['wells']:4d} wells {c['by_partner']}")


if __name__ == "__main__":
    main()
