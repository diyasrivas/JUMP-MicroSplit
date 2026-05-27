#!/usr/bin/env python3
"""
Training dataset generation for cpg0036-EU-OS-bioactives MicroSplit.

Reads Cell Painting plates (EU-OPENSCREEN bioactive compound screen) from local
disk, samples FOVs stratified by plate, and writes a MicroSplit dataset.

Dataset details (Wolff et al., iScience 2025):
  - 2,464 EU-OPENSCREEN Bioactive compounds at 10 μM
  - Two cell lines: Hep G2 and U-2 OS (train separately!)
  - Imaged on Opera Phenix (spinning disk, 20× water, NA 1.0, 2× binning)
  - 9 fields per well, 384-well plates × 7 plates × 4 replicates
  - 4 imaging channels (NOT 5) — two channels each contain 2 co-stained dyes:

    Channel     Wavelength   Stains
    ──────────  ──────────   ─────────────────────────────────────
    Mito (ch1)  650-760 nm   MitoTracker Deep Red (mitochondria)
    Actin_AGP   570-630 nm   WGA Alexa 555 (Golgi/PM) +
    (ch2)                    Phalloidin Alexa 568 (F-actin)
    ER_RNA      500-550 nm   Concanavalin A Alexa 488 (ER) +
    (ch3)                    SYTO 14 (nucleolar/cytoplasmic RNA)
    DNA (ch4)   435-480 nm   Hoechst 33342 (nucleus)
    ──────────  ──────────   ─────────────────────────────────────

  The Actin_AGP and ER_RNA channels are spectrally mixed — this is precisely
  the MicroSplit unmixing task for this dataset.

  NOTE: Channel indices (ch1–ch4) follow the Opera Phenix convention of longest
        to shortest emission wavelength. Verify against actual data files before
        running if in doubt.

Output layout:
  training_dataset_cpg0036_{cell_line}/
      combined/        float32 sum of DNA+ER_RNA+Actin_AGP+Mito
      DNA/             uint16 per-channel images
      ER_RNA/
      Actin_AGP/
      Mito/
      metadata.csv

Usage:
    python 1_datasets.py --cell_line HepG2
    python 1_datasets.py --cell_line U2OS \\
        --data_dir /project/cell_paint_mono/cpg0036-EU-OS-bioactives \\
        --samples 3500 --seed 42
"""

import argparse
import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np

from microsplit_reproducibility.workflows.cellpainting import (
    get_available_sites,
    build_dataset_from_samples,
)

# ---------------------------------------------------------------------------
# Channel configuration (4-channel Opera Phenix, EU-OPENSCREEN Cell Painting)
# ch ordering: longest to shortest emission wavelength (same convention as cpg0000)
# ---------------------------------------------------------------------------

CHANNEL_NAMES   = ["DNA", "ER_RNA", "Actin_AGP", "Mito"]
CHANNEL_MAPPING = {
    "DNA":       4,   # Hoechst 33342          (435-480 nm)
    "ER_RNA":    3,   # ConA Alexa 488 + SYTO14 (500-550 nm)  ← spectrally mixed
    "Actin_AGP": 2,   # WGA Alexa 555 + Phalloidin Alexa 568 (570-630 nm) ← mixed
    "Mito":      1,   # MitoTracker Deep Red   (650-760 nm)
}

VALID_CELL_LINES = ["HepG2", "U2OS"]


# ---------------------------------------------------------------------------
# Plate discovery
# ---------------------------------------------------------------------------

def find_available_plates(data_dir: Path) -> List[Tuple[str, str, Path]]:
    """Discover all plates under data_dir/{batch}/images/{plate}/Images/.

    Returns list of (batch, plate_dir_name, images_path).
    """
    available = []
    for images_path in sorted(data_dir.glob("*/images/*/Images")):
        batch = images_path.parts[-4]   # {batch}/images/{plate}/Images
        plate = images_path.parts[-2]
        available.append((batch, plate, images_path))
    if not available:
        print(
            f"WARNING: No plates found using */images/*/Images pattern in {data_dir}",
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
    samples_per_plate = max(1, target_samples // len(plates))

    for batch, plate, images_path in plates:
        all_sites = get_available_sites(images_path)
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
        description="Build cpg0036-EU-OS-bioactives training dataset for MicroSplit"
    )
    parser.add_argument(
        "--cell_line",
        required=True,
        choices=VALID_CELL_LINES,
        help="Cell line to build dataset for (HepG2 or U2OS). "
             "Point --data_dir at a directory containing only that cell line's plates. "
             "The FMP site imaged both HepG2 and U2OS; other sites (IMTM, MEDINA, USC) "
             "imaged HepG2 only.",
    )
    parser.add_argument(
        "--data_dir",
        default="/project/cell_paint_mono/cpg0036-EU-OS-bioactives",
        help="Root containing {batch}/images/{plate}/Images/  "
             "(should contain only the target cell line's plates, or use "
             "--plate_filter to restrict by plate-name substring)",
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        help="Output directory (default: "
             "/project/cell_paint_mono/training_datasets/training_dataset_cpg0036_{cell_line})",
    )
    parser.add_argument("--samples", type=int, default=3500,
                        help="Target total FOVs to sample")
    parser.add_argument("--seed",    type=int, default=42)
    parser.add_argument(
        "--plate_filter",
        nargs="*",
        default=[],
        help="Optional list of substrings; only plates whose directory name "
             "contains at least one match are included. Use to isolate one cell "
             "line when both are under the same data_dir.",
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    output_dir = Path(
        args.output_dir
        or f"/project/cell_paint_mono/training_datasets/training_dataset_cpg0036_{args.cell_line}"
    )

    print(f"\nCell line:    {args.cell_line}")
    print(f"Channels:     {CHANNEL_NAMES}")
    print(f"Mapping:      {CHANNEL_MAPPING}")
    print(f"Scanning plates in {data_dir} ...")
    plates = find_available_plates(data_dir)

    if args.plate_filter:
        plates = [
            (b, p, ip) for b, p, ip in plates
            if any(f in p for f in args.plate_filter)
        ]
        print(f"After plate_filter ({args.plate_filter}): {len(plates)} plates")

    if not plates:
        print("ERROR: No plates found on disk. Check --data_dir and --plate_filter.",
              file=sys.stderr)
        sys.exit(1)
    print(f"Found {len(plates)} plates.")

    print(f"\nSampling {args.samples} training FOVs ...")
    train_samples = sample_fovs(plates, args.samples, args.seed)
    print(f"Total training samples: {len(train_samples)}")

    extra_metadata = [{"cell_line": args.cell_line} for _ in train_samples]

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
