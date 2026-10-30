#!/usr/bin/env python3
"""
Step 1: build the MicroSplit training dataset for cpg0036-EU-OS-bioactives from
the training rows of the split manifest written by 0_split.py.

Every FOV is centre-cropped to 1000 x 1000 pixels, the size of the IMTM images,
so that FOVs from all partner sites share one shape.

Writes:
  training_dataset_cpg0036_allsites_{cell_line}/
      combined/                        float32 average of the four channels
      DNA/ ER_RNA/ Actin_AGP/ Mito/    uint16 individual channels
      metadata.csv                     per-FOV metadata
      split_manifest_used.csv          copy of the manifest for provenance

Usage:
    python 1_datasets.py --cell_line HepG2
    python 1_datasets.py --cell_line U2OS --manifest /path/to/split_manifest.csv
"""

import argparse
from pathlib import Path

from microsplit_reproducibility.workflows.cellpainting import build_dataset_from_manifest

CHANNELS = ["DNA", "ER_RNA", "Actin_AGP", "Mito"]
CROP_SIZE = 1000

DEFAULT_SPLITS_DIR = "/project/cell_paint_mono/splits/cpg0036-EU-OS-bioactives"
DEFAULT_OUT_BASE = "/project/cell_paint_mono/training_datasets"


def main() -> None:
    ap = argparse.ArgumentParser(description="Build the cpg0036 training dataset from the split manifest")
    ap.add_argument("--cell_line", required=True, choices=["HepG2", "U2OS"])
    ap.add_argument("--manifest", default=None,
                    help="split_manifest.csv (default: {splits_dir}/{cell_line}/split_manifest.csv)")
    ap.add_argument("--splits_dir", default=DEFAULT_SPLITS_DIR)
    ap.add_argument("--output_dir", default=None,
                    help="default: {out_base}/training_dataset_cpg0036_allsites_{cell_line}")
    ap.add_argument("--out_base", default=DEFAULT_OUT_BASE)
    ap.add_argument("--max_fovs", type=int, default=0, help="only the first N training rows (testing)")
    args = ap.parse_args()

    manifest = Path(args.manifest or f"{args.splits_dir}/{args.cell_line}/split_manifest.csv")
    output_dir = Path(
        args.output_dir or f"{args.out_base}/training_dataset_cpg0036_allsites_{args.cell_line}"
    )
    print(f"Cell line: {args.cell_line}\nManifest:  {manifest}\nOutput:    {output_dir}")
    build_dataset_from_manifest(
        manifest, args.cell_line, CHANNELS, output_dir,
        crop_size=CROP_SIZE, max_fovs=args.max_fovs,
    )


if __name__ == "__main__":
    main()
