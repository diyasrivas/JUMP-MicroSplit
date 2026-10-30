"""
Split-manifest utilities shared by the Cell Painting experiments.

A split manifest is a CSV with one row per (plate, well, site, role) produced by
each experiment's ``0_split.py``. It is the single source of truth for which FOVs
are used for training and which are predicted (test). Prediction scripts read
it to select FOVs, append per-FOV metadata as they go (so a killed job leaves a
usable partial result), and write CellProfiler ``load_data`` CSVs at the end.

Manifests of datasets whose files do not follow the Opera Phenix naming list the
image of every channel explicitly, in one ``path_{channel}`` column per channel.
"""

import csv
import os
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence

import pandas as pd

TRAIN_ROLE = "train"
TEST_ROLES = ("test_perturbation", "test_negcon")
PATH_PREFIX = "path_"


def channel_path_column(channel: str) -> str:
    """Name of the manifest column holding the image path of ``channel``."""
    return f"{PATH_PREFIX}{channel}"


def row_channel_paths(row: Mapping, channel_names: Sequence[str]) -> Dict[str, str]:
    """Channel name -> image path for one manifest row."""
    return {ch: row[channel_path_column(ch)] for ch in channel_names}


def load_manifest(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, dtype={"well": str, "broad_sample": str, "gene": str,
                                  "control_type": str, "pert_type": str})
    for c in ("broad_sample", "gene", "control_type", "pert_type"):
        if c in df.columns:
            df[c] = df[c].fillna("")
    return df


def manifest_rows_for_plate(
    df: pd.DataFrame,
    plate: str,
    roles: Sequence[str] = TEST_ROLES,
    plate_column: str = "plate_dir",
) -> pd.DataFrame:
    """Rows of one plate restricted to the given roles, one per FOV."""
    sub = df[(df[plate_column] == plate) & (df.role.isin(roles))]
    return sub.drop_duplicates(subset=["well", "site"]).sort_values(["well", "site"]).reset_index(drop=True)


def append_metadata_row(csv_path: str, row: Dict) -> None:
    """Append one row to a CSV, writing the header if the file is new.

    Column order is fixed by the first row written; later rows with extra keys
    are truncated to the header, missing keys are left empty.
    """
    exists = os.path.isfile(csv_path) and os.path.getsize(csv_path) > 0
    if exists:
        with open(csv_path, newline="") as f:
            header = next(csv.reader(f))
    else:
        header = list(row.keys())
    with open(csv_path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=header, extrasaction="ignore")
        if not exists:
            w.writeheader()
        w.writerow({k: row.get(k, "") for k in header})


def completed_fovs(csv_path: str) -> set:
    """(well, site) pairs already present in a per-FOV metadata.csv."""
    if not os.path.isfile(csv_path) or os.path.getsize(csv_path) == 0:
        return set()
    df = pd.read_csv(csv_path, usecols=["well", "site"])
    return set(zip(df.well.astype(str), df.site.astype(int)))


def write_cellprofiler_load_data(
    rows: Iterable[Dict],
    out_csv: str,
    barcode: str,
    raw_images_dir: str,
    channel_mapping: Dict[str, int],
    illum_dir: Optional[str] = None,
    predicted_dir: Optional[str] = None,
    extra_raw_channels: Optional[Dict[str, int]] = None,
    metadata_keys: Sequence[str] = (),
) -> int:
    """Write a CellProfiler LoadData CSV in the JUMP ``load_data_with_illum`` layout.

    For every row (needs ``well``, ``site``, ``image_prefix``) writes
    ``URL_Orig{Channel}`` for each channel in ``channel_mapping`` — pointing at
    the MMSE prediction ``predicted_dir/mmse/mmse_{well}_s{site:02d}_{Channel}.tif``
    when ``predicted_dir`` is given, otherwise at the raw tiff — plus
    ``URL_Orig{Channel}`` for ``extra_raw_channels`` (e.g. brightfield, always
    raw) and ``URL_Illum{Channel}`` for every channel when ``illum_dir`` is given
    (``{illum_dir}/{barcode}_Illum{Channel}.npy``). ``Metadata_Plate``,
    ``Metadata_Well``, ``Metadata_Site`` are always written; additional row keys
    listed in ``metadata_keys`` become ``Metadata_{key}`` columns.

    Returns the number of rows written.
    """
    extra_raw_channels = extra_raw_channels or {}
    all_channels = list(channel_mapping) + list(extra_raw_channels)
    cols: List[str] = []
    if illum_dir:
        cols += [f"URL_Illum{ch}" for ch in all_channels]
    cols += [f"URL_Orig{ch}" for ch in all_channels]
    cols += ["Metadata_Plate", "Metadata_Well", "Metadata_Site"]
    cols += [f"Metadata_{k}" for k in metadata_keys]

    n = 0
    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            well, site, prefix = str(r["well"]), int(r["site"]), r["image_prefix"]
            rec = {}
            if illum_dir:
                for ch in all_channels:
                    rec[f"URL_Illum{ch}"] = os.path.join(illum_dir, f"{barcode}_Illum{ch}.npy")
            for ch, idx in channel_mapping.items():
                if predicted_dir:
                    rec[f"URL_Orig{ch}"] = os.path.join(predicted_dir, "mmse", f"mmse_{well}_s{site:02d}_{ch}.tif")
                else:
                    rec[f"URL_Orig{ch}"] = os.path.join(raw_images_dir, f"{prefix}-ch{idx}sk1fk1fl1.tiff")
            for ch, idx in extra_raw_channels.items():
                rec[f"URL_Orig{ch}"] = os.path.join(raw_images_dir, f"{prefix}-ch{idx}sk1fk1fl1.tiff")
            rec["Metadata_Plate"] = barcode
            rec["Metadata_Well"] = well
            rec["Metadata_Site"] = site
            for k in metadata_keys:
                rec[f"Metadata_{k}"] = r.get(k, "")
            w.writerow(rec)
            n += 1
    return n
