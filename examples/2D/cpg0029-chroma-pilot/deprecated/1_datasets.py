#!/usr/bin/env python3
"""
Training dataset generation for cpg0029-chroma-pilot MicroSplit.

Reads fixed Cell Painting plates (7 plates, U2OS cells) from local disk,
samples FOVs stratified by dye_condition, and writes a MicroSplit dataset:

  training_dataset_cpg0029/
      combined/        float32 sum of DNA+RNA+ER+AGP+Mito
      DNA/ RNA/ ER/ AGP/ Mito/  uint16 individual channels
      metadata.csv
      test_wells.csv   held-out wells (30% per plate, for evaluation)

Channel mapping (cpg0029-chroma-pilot, fixed plates, 8-ch Opera Phenix):
  ch1=Phase, ch2=Brightfield, ch3=RNA, ch4=ER, ch5=AGP,
  ch6=Mito,  ch7=DNA,         ch8=Actin (PhenoVue 400LS)

MicroSplit targets: DNA(ch7), RNA(ch3), ER(ch4), AGP(ch5), Mito(ch6)

Usage:
    python 1_datasets.py
    python 1_datasets.py --data_dir /project/cell_paint_mono/cpg0029-chroma-pilot/images
    python 1_datasets.py --samples 3500 --holdout_frac 0.3
"""

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from microsplit_reproducibility.workflows.cellpainting import (
    get_available_sites,
    build_dataset_from_samples,
)

# ---------------------------------------------------------------------------
# Channel configuration
# ---------------------------------------------------------------------------

CHANNEL_NAMES   = ["DNA", "RNA", "ER", "AGP", "Mito"]
CHANNEL_MAPPING = {"DNA": 7, "RNA": 3, "ER": 4, "AGP": 5, "Mito": 6}

# ---------------------------------------------------------------------------
# Fixed plate manifest (cpg0029-chroma-pilot)
# ---------------------------------------------------------------------------

FIXED_PLATES = [
    ("2023_05_15_Batch1", "BR00122246__2023-04-01T03_13_00-Measurement1"),
    ("2023_05_15_Batch1", "BR00122250__2023-04-01T05_43_06-Measurement1"),
    ("2023_05_17_Batch2", "BR00122247__2023-04-01T08_12_58-Measurement1"),
    ("2023_05_17_Batch3", "BR00122248__2023-04-01T10_42_56-Measurement1"),
    ("2023_08_02_Batch4", "BR00122245__2023-03-25T08_08_05-Measurement1"),
    ("2023_08_02_Batch5", "BR00122249__2023-03-25T00_14_17-Measurement2"),
    ("2025_02_13_Batch9", "BR00122244__2023-04-01T13_12_51-Measurement1"),
]

# Dye condition from plate barcode
DYE_CONDITION_MAP = {
    "BR00122244": "standard_cp",
    "BR00122246": "standard_cp",
    "BR00122247": "alt_mito",
    "BR00122248": "post_chromalive",
    "BR00122249": "post_chromalive",
    "BR00122250": "standard_cp",
    "BR00122245": "post_chromalive",
}


# ---------------------------------------------------------------------------
# Plate discovery
# ---------------------------------------------------------------------------

def find_available_plates(
    data_dir: Path,
) -> List[Tuple[str, str, Path]]:
    """Return list of (batch, plate_dir_name, images_path) for plates on disk."""
    available = []
    for batch, plate in FIXED_PLATES:
        images_path = data_dir / batch / "images" / plate / "Images"
        if images_path.exists():
            available.append((batch, plate, images_path))
        else:
            print(f"  SKIP (not found): {images_path}")
    return available


# ---------------------------------------------------------------------------
# Stratified sampling (by dye condition, with well holdout)
# ---------------------------------------------------------------------------

def sample_fovs(
    plates: List[Tuple[str, str, Path]],
    target_samples: int,
    holdout_frac: float,
    seed: int,
) -> Tuple[List[Tuple[str, str, str, int]], Dict[str, List[str]]]:
    """Sample FOVs uniformly across plates, holding out a fraction of wells.

    Returns
    -------
    train_samples : list of (batch, plate, well, site)
    test_wells    : dict mapping plate_dir -> list of held-out wells
    """
    rng = np.random.default_rng(seed)
    train_samples: List[Tuple[str, str, str, int]] = []
    test_wells: Dict[str, List[str]] = {}
    samples_per_plate = max(1, target_samples // len(plates))

    for batch, plate, images_path in plates:
        all_sites = get_available_sites(images_path)
        if not all_sites:
            print(f"  WARNING: No sites found in {images_path}, skipping")
            continue

        wells    = sorted(set(w for w, _ in all_sites))
        rng.shuffle(wells)
        n_holdout   = max(1, int(len(wells) * holdout_frac))
        holdout_set = set(wells[:n_holdout])
        train_wells = set(wells[n_holdout:])

        test_wells[plate] = sorted(holdout_set)

        train_fovs = [(w, s) for w, s in all_sites if w in train_wells]
        if not train_fovs:
            print(f"  WARNING: No training FOVs for {plate} after holdout split")
            continue

        n_select = min(samples_per_plate, len(train_fovs))
        idx = rng.choice(len(train_fovs), size=n_select, replace=False)
        for i in idx:
            well, site = train_fovs[int(i)]
            train_samples.append((batch, plate, well, site))

        barcode = plate.split("__")[0]
        dye     = DYE_CONDITION_MAP.get(barcode, "unknown")
        print(
            f"  {barcode} ({dye}): "
            f"{len(train_wells)} train wells, {len(holdout_set)} holdout, "
            f"{n_select} FOVs sampled"
        )

    return train_samples, test_wells


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Build cpg0029-chroma-pilot training dataset for MicroSplit"
    )
    parser.add_argument(
        "--data_dir",
        default="/project/cell_paint_mono/cpg0029-chroma-pilot/images",
        help="Root containing {batch}/images/{plate}/Images/",
    )
    parser.add_argument(
        "--output_dir",
        default="/project/cell_paint_mono/training_datasets/training_dataset_cpg0029",
    )
    parser.add_argument("--samples",      type=int,   default=3500)
    parser.add_argument("--holdout_frac", type=float, default=0.3,
                        help="Fraction of wells held out per plate for evaluation")
    parser.add_argument("--seed",         type=int,   default=42)
    args = parser.parse_args()

    data_dir   = Path(args.data_dir)
    output_dir = Path(args.output_dir)

    print(f"\nScanning plates in {data_dir} ...")
    plates = find_available_plates(data_dir)
    if not plates:
        print("ERROR: No plates found on disk. Check --data_dir.", file=sys.stderr)
        sys.exit(1)
    print(f"Found {len(plates)} plates.")

    print(f"\nSampling {args.samples} training FOVs (holdout={args.holdout_frac:.0%}/plate) ...")
    train_samples, test_wells = sample_fovs(
        plates, args.samples, args.holdout_frac, args.seed
    )
    print(f"Total training samples: {len(train_samples)}")

    # Build extra per-sample metadata (dye condition)
    plate_batch_lookup = {plate: batch for batch, plate, _ in plates}
    extra = []
    for batch, plate, well, site in train_samples:
        barcode = plate.split("__")[0]
        extra.append({"dye_condition": DYE_CONDITION_MAP.get(barcode, "unknown")})

    build_dataset_from_samples(
        samples=train_samples,
        data_dir=data_dir.parent,   # data_dir already points to {root}/images/
        output_dir=output_dir,
        channel_names=CHANNEL_NAMES,
        channel_mapping=CHANNEL_MAPPING,
        extra_metadata=extra,
    )

    # Write test_wells.csv
    test_path = output_dir / "test_wells.csv"
    with open(test_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["plate", "well"])
        for plate, wells in sorted(test_wells.items()):
            for w in wells:
                writer.writerow([plate, w])
    print(f"  test_wells.csv -> {test_path}")

    # Summary by dye condition
    print("\nDye condition breakdown:")
    by_dye: Dict[str, int] = defaultdict(int)
    for e in extra:
        by_dye[e["dye_condition"]] += 1
    for cond, count in sorted(by_dye.items()):
        print(f"  {cond}: {count}")


if __name__ == "__main__":
    main()
