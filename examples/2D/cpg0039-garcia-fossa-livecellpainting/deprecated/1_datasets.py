#!/usr/bin/env python3
"""
Training dataset generation for cpg0039-garcia-fossa-livecellpainting MicroSplit.

Reads Live Cell Painting plates from local disk, samples FOVs stratified by plate,
and writes a MicroSplit dataset.

Dataset details (Garcia-Fossa et al. 2025):
  - Live Cell Painting using a single Acridine Orange (AO) metachromatic dye
  - Only 2 fluorescence channels:
      AO_Red   (ch1, PI filter, Ex/Em 531/647 nm)   → acidic organelles
      AO_Green (ch2, GFP filter, Ex/Em 469/525 nm)  → DNA + RNA
  - Four cell lines: Huh_7, MCF_7, PNT1A, PC_3 (train one model per cell line)
  - Imaged on Cytation 5 BioTek (NOT Opera Phenix), 20× NA 0.45 objective
  - 96-well plates, 16 sites per well, 1224 × 904 pixels, 16-bit TIFFs

Channel mapping:
  AO_Red   = ch1   (531/647 nm, longer wavelength → index 1, consistent with
                    cpg0000/cpg0036 longest-first convention)
  AO_Green = ch2   (469/525 nm, shorter wavelength → index 2)

  *** IMPORTANT: Verify these channel indices against the actual cpg0039 TIFF
      files before running. The Cytation 5 BioTek file-naming and channel-index
      convention may differ from the Opera Phenix convention used in other
      experiments. Inspect a sample TIFF directory and adjust CHANNEL_MAPPING
      and FILE_PATTERN below as needed. ***

File naming (Cytation 5 / Cell Painting Gallery):
  The Cytation 5 BioTek Gen5 software uses a different naming convention than
  Opera Phenix. cpg0039 files are expected in one of these formats:
    {row:02d}{col:02d}_01_1_1_{channel}.tif         (Gen5 flat export)
    {row:02d}{col:02d}_s{site}_{channel}.tif         (common variant)
  Adjust discover_fovs_cpg0039() below if the actual format differs.

Output layout:
  training_dataset_cpg0039_{cell_line}/
      combined/        float32 sum of AO_Green + AO_Red
      AO_Green/        uint16 individual channel images
      AO_Red/
      metadata.csv

Usage:
    python 1_datasets.py --cell_line Huh_7
    python 1_datasets.py --cell_line MCF_7 \\
        --data_dir /project/cell_paint_mono/cpg0039-garcia-fossa-livecellpainting \\
        --samples 2500 --seed 42
"""

import argparse
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from microsplit_reproducibility.workflows.cellpainting import (
    build_dataset_from_samples,
)

# ---------------------------------------------------------------------------
# Channel configuration  (2-channel Live Cell Painting, Acridine Orange only)
# ---------------------------------------------------------------------------

CHANNEL_NAMES   = ["AO_Green", "AO_Red"]
CHANNEL_MAPPING = {
    "AO_Green": 2,   # GFP filter Ex/Em 469/525 nm  (shorter wavelength → ch2)
    "AO_Red":   1,   # PI filter  Ex/Em 531/647 nm  (longer wavelength  → ch1)
}

VALID_CELL_LINES = ["Huh_7", "MCF_7", "PNT1A", "PC_3"]


# ---------------------------------------------------------------------------
# Custom FOV discovery for Cytation 5 / cpg0039
# ---------------------------------------------------------------------------
# The BioTek Cytation 5 Gen5 software names files differently from Opera Phenix.
# Adjust the patterns below to match the actual cpg0039 TIFF filenames.
# Common Cytation 5 patterns seen in Cell Painting Gallery depositions:
#   r{row:02d}c{col:02d}f{site:02d}p01-ch{ch}sk1fk1fl1.tiff  (standardised CPG)
#   {row:02d}{col:02d}_01_{site}_{ch}.tif                      (Gen5 flat)
# ---------------------------------------------------------------------------

def discover_fovs_cpg0039(images_dir: Path) -> List[Tuple[str, int]]:
    """Return list of (well, site) pairs found in images_dir.

    Tries the standardised Cell Painting Gallery Opera-Phenix-style pattern
    first (r{row:02d}c{col:02d}f{site}...), then falls back to a Cytation 5
    Gen5 flat-export pattern.  Adjust if neither matches your data.
    """
    seen: Dict[Tuple[str, int], None] = {}

    # Pattern 1: standardised CPG naming (Opera Phenix style, used even for
    # non-Opera data when re-uploaded to CPG)
    for f in sorted(images_dir.glob("r[0-9][0-9]c[0-9][0-9]f[0-9]*-ch*.tif*")):
        m = re.match(r"r(\d+)c(\d+)f(\d+)", f.name)
        if m:
            row, col, site = int(m.group(1)), int(m.group(2)), int(m.group(3))
            well = f"r{row:02d}c{col:02d}"
            seen[(well, site)] = None

    if seen:
        return list(seen.keys())

    # Pattern 2: Cytation 5 Gen5 flat export — e.g. 0101_01_1_1_GFP.tif
    # Columns 1-12, rows A-H → numeric row/col
    for f in sorted(images_dir.glob("[0-9][0-9][0-9][0-9]_*.tif*")):
        m = re.match(r"(\d{2})(\d{2})_0*(\d+)_", f.name)
        if m:
            row, col, site = int(m.group(1)), int(m.group(2)), int(m.group(3))
            # Convert numeric row to letter (1→A, 2→B, …)
            row_letter = chr(ord('A') + row - 1)
            well = f"{row_letter}{col:02d}"
            seen[(well, site)] = None

    return list(seen.keys())


def get_available_sites_cpg0039(images_dir: Path) -> List[Tuple[str, int]]:
    """Wrapper matching the signature of the library function."""
    return discover_fovs_cpg0039(images_dir)


# ---------------------------------------------------------------------------
# Plate discovery
# ---------------------------------------------------------------------------

def find_available_plates(data_dir: Path) -> List[Tuple[str, str, Path]]:
    """Discover all plates under data_dir/{batch}/images/{plate}/Images/.

    Returns list of (batch, plate_dir_name, images_path).
    """
    available = []
    for images_path in sorted(data_dir.glob("*/images/*/Images")):
        batch = images_path.parts[-4]
        plate = images_path.parts[-2]
        available.append((batch, plate, images_path))

    # Also try flat structure: data_dir/{plate}/Images
    if not available:
        for images_path in sorted(data_dir.glob("*/Images")):
            plate = images_path.parts[-2]
            available.append((".", plate, images_path))

    if not available:
        print(
            f"WARNING: No plates found in {data_dir}",
            file=sys.stderr,
        )
    return available


# ---------------------------------------------------------------------------
# Stratified sampling (by plate)
# ---------------------------------------------------------------------------

def sample_fovs(
    plates: List[Tuple[str, str, Path]],
    target_samples: int,
    seed: int,
) -> List[Tuple[str, str, str, int]]:
    """Sample FOVs uniformly across plates.

    Returns list of (batch, plate, well, site).
    """
    rng = np.random.default_rng(seed)
    train_samples: List[Tuple[str, str, str, int]] = []
    samples_per_plate = max(1, target_samples // max(len(plates), 1))

    for batch, plate, images_path in plates:
        all_sites = get_available_sites_cpg0039(images_path)
        if not all_sites:
            print(f"  WARNING: No sites found in {images_path}, skipping")
            continue

        n_select = min(samples_per_plate, len(all_sites))
        idx = rng.choice(len(all_sites), size=n_select, replace=False)
        for i in idx:
            well, site = all_sites[int(i)]
            train_samples.append((batch, plate, well, site))

        print(f"  {plate}: {len(all_sites)} total FOVs, {n_select} sampled")

    return train_samples


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Build cpg0039-garcia-fossa-livecellpainting training dataset for MicroSplit"
    )
    parser.add_argument(
        "--cell_line",
        required=True,
        choices=VALID_CELL_LINES,
        help="Cell line to build dataset for (Huh_7 | MCF_7 | PNT1A | PC_3). "
             "Each cell line must be trained separately. Point --data_dir at a "
             "directory containing only that cell line's plates.",
    )
    parser.add_argument(
        "--data_dir",
        default="/project/cell_paint_mono/cpg0039-garcia-fossa-livecellpainting",
        help="Root containing plates for this cell line. "
             "Expected structure: {batch}/images/{plate}/Images/ "
             "or {plate}/Images/",
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        help="Output directory "
             "(default: /project/cell_paint_mono/training_datasets/"
             "training_dataset_cpg0039_{cell_line})",
    )
    parser.add_argument("--samples", type=int, default=2500,
                        help="Target total FOVs to sample (default: 2500 — "
                             "96-well plates are smaller than 384-well)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    output_dir = Path(
        args.output_dir
        or f"/project/cell_paint_mono/training_datasets/training_dataset_cpg0039_{args.cell_line}"
    )

    print(f"\nCell line:    {args.cell_line}")
    print(f"Channels:     {CHANNEL_NAMES}")
    print(f"Mapping:      {CHANNEL_MAPPING}")
    print("  *** Verify channel indices against actual TIFF files! ***")
    print(f"Scanning plates in {data_dir} ...")

    plates = find_available_plates(data_dir)
    if not plates:
        print("ERROR: No plates found. Check --data_dir.", file=sys.stderr)
        sys.exit(1)
    print(f"Found {len(plates)} plates.")

    print(f"\nSampling {args.samples} training FOVs ...")
    train_samples = sample_fovs(plates, args.samples, args.seed)
    print(f"Total training samples: {len(train_samples)}")

    if not train_samples:
        print("ERROR: No FOVs found. Check file naming pattern in "
              "discover_fovs_cpg0039().", file=sys.stderr)
        sys.exit(1)

    extra_metadata = [{"cell_line": args.cell_line} for _ in train_samples]

    from microsplit_reproducibility.workflows.cellpainting import build_dataset_from_samples
    build_dataset_from_samples(
        samples=train_samples,
        data_dir=data_dir,
        output_dir=output_dir,
        channel_names=CHANNEL_NAMES,
        channel_mapping=CHANNEL_MAPPING,
        extra_metadata=extra_metadata,
    )


if __name__ == "__main__":
    main()
