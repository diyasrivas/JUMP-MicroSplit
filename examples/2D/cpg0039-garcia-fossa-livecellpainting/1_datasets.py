#!/usr/bin/env python3
"""
Step 1: build the MicroSplit training dataset for cpg0039-garcia-fossa-livecellpainting
from the training rows of the split manifest written by 0_split.py.

Writes:
  training_dataset_cpg0039_Huh7/
      combined/                float32 average of the two channels
      AO_Green/ AO_Red/        uint16 individual channels
      metadata.csv             per-FOV metadata
      split_manifest_used.csv  copy of the manifest for provenance

Usage:
    python 1_datasets.py
    python 1_datasets.py --manifest /path/to/split_manifest.csv
"""

import argparse
from pathlib import Path

from microsplit_reproducibility.workflows.cellpainting import build_dataset_from_manifest

CHANNELS = ["AO_Green", "AO_Red"]
CELL_LINE = "Huh7"

DEFAULT_SPLITS_DIR = "/project/cell_paint_mono/splits/cpg0039-garcia-fossa-livecellpainting"
DEFAULT_OUT_BASE = "/project/cell_paint_mono/training_datasets"


def main() -> None:
    ap = argparse.ArgumentParser(description="Build the cpg0039 training dataset from the split manifest")
    ap.add_argument("--manifest", default=None,
                    help="split_manifest.csv (default: {splits_dir}/Huh7/split_manifest.csv)")
    ap.add_argument("--splits_dir", default=DEFAULT_SPLITS_DIR)
    ap.add_argument("--output_dir", default=None,
                    help="default: {out_base}/training_dataset_cpg0039_Huh7")
    ap.add_argument("--out_base", default=DEFAULT_OUT_BASE)
    ap.add_argument("--max_fovs", type=int, default=0, help="only the first N training rows (testing)")
    args = ap.parse_args()

    manifest = Path(args.manifest or f"{args.splits_dir}/{CELL_LINE}/split_manifest.csv")
    output_dir = Path(args.output_dir or f"{args.out_base}/training_dataset_cpg0039_{CELL_LINE}")
    print(f"Manifest: {manifest}\nOutput:   {output_dir}")
    build_dataset_from_manifest(manifest, CELL_LINE, CHANNELS, output_dir, max_fovs=args.max_fovs)


if __name__ == "__main__":
    main()
