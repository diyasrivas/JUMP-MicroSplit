#!/usr/bin/env python3
"""
Step 1 — build the MicroSplit training dataset for cpg0000-jump-pilot from the
split manifest written by 0_split.py (rows with role == "train").

Writes:
  training_dataset_cpg0000_{design}_{cell_line}/
      combined/        float32 average of DNA+RNA+ER+AGP+Mito
      DNA/ RNA/ ER/ AGP/ Mito/   uint16 individual channels
      metadata.csv     per-FOV metadata (plate, well, site, broad_sample, gene, modality, timepoint)
      split_manifest_used.csv    copy of the manifest for provenance

Channel mapping (cpg0000-jump-pilot, Opera Phenix):
  ch1=Mito, ch2=AGP, ch3=RNA, ch4=ER, ch5=DNA

Usage:
    python 1_datasets.py --design gene_subset --cell_line A549
    python 1_datasets.py --design random_wells --cell_line U2OS
"""

import argparse
import shutil
from pathlib import Path

import pandas as pd

from microsplit_reproducibility.workflows.cellpainting import build_dataset_from_samples

CHANNEL_NAMES   = ["DNA", "RNA", "ER", "AGP", "Mito"]
CHANNEL_MAPPING = {"DNA": 5, "RNA": 3, "ER": 4, "AGP": 2, "Mito": 1}

DEFAULT_DATA_DIR   = "/project/cell_paint_mono"
DEFAULT_SPLITS_DIR = "/project/cell_paint_mono/splits/cpg0000-jump-pilot"
DEFAULT_OUT_BASE   = "/project/cell_paint_mono/training_datasets"


def main():
    ap = argparse.ArgumentParser(description="Build cpg0000 training dataset from the split manifest")
    ap.add_argument("--design", required=True, choices=["gene_subset", "random_wells"])
    ap.add_argument("--cell_line", required=True, choices=["A549", "U2OS"])
    ap.add_argument("--manifest", default=None,
                    help="split_manifest.csv (default: {splits_dir}/{design}/{cell_line}/split_manifest.csv)")
    ap.add_argument("--splits_dir", default=DEFAULT_SPLITS_DIR)
    ap.add_argument("--data_dir", default=DEFAULT_DATA_DIR,
                    help="Root containing {batch}/images/{plate_dir}/Images/")
    ap.add_argument("--output_dir", default=None,
                    help="default: {out_base}/training_dataset_cpg0000_{design}_{cell_line}")
    ap.add_argument("--out_base", default=DEFAULT_OUT_BASE)
    ap.add_argument("--max_fovs", type=int, default=0, help="debug: only the first N training rows")
    args = ap.parse_args()

    manifest = Path(args.manifest or f"{args.splits_dir}/{args.design}/{args.cell_line}/split_manifest.csv")
    output_dir = Path(
        args.output_dir or f"{args.out_base}/training_dataset_cpg0000_{args.design}_{args.cell_line}"
    )
    if not manifest.exists():
        raise FileNotFoundError(f"{manifest} not found — run 0_split.py first")

    man = pd.read_csv(manifest)
    train = man[(man.role == "train") & (man.cell_line == args.cell_line)].reset_index(drop=True)
    if args.max_fovs > 0:
        train = train.head(args.max_fovs)
    if train.empty:
        raise ValueError("No training rows in manifest")

    print(f"Cell line:  {args.cell_line}")
    print(f"Manifest:   {manifest}")
    print(f"Output:     {output_dir}")
    print(f"Train FOVs: {len(train)} on {train.plate_dir.nunique()} plates")
    print(train.groupby(["modality", "timepoint"]).size().to_string())

    samples = [(r.batch, r.plate_dir, r.well, int(r.site)) for r in train.itertuples()]
    extra = [{
        "barcode": r.barcode, "modality": r.modality, "timepoint": r.timepoint,
        "time_h": int(r.time_h), "broad_sample": r.broad_sample if isinstance(r.broad_sample, str) else "",
        "gene": r.gene if isinstance(r.gene, str) else "",
        "control_type": r.control_type if isinstance(r.control_type, str) else "",
        "role": r.role, "cell_line": r.cell_line,
    } for r in train.itertuples()]

    build_dataset_from_samples(
        samples=samples,
        data_dir=Path(args.data_dir),
        output_dir=output_dir,
        channel_names=CHANNEL_NAMES,
        channel_mapping=CHANNEL_MAPPING,
        extra_metadata=extra,
    )
    shutil.copy(manifest, output_dir / "split_manifest_used.csv")
    print(f"Manifest copied to {output_dir / 'split_manifest_used.csv'}")


if __name__ == "__main__":
    main()
