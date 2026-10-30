#!/usr/bin/env python3
"""
Step 4: MicroSplit prediction for cpg0036-EU-OS-bioactives, one plate per job.

Predicts the test FOVs of one plate of the split manifest at full image size,
MMSE only. Each FOV is written as soon as it finishes and one row is appended
to metadata.csv, so rerunning a plate resumes where it stopped.

Output:
  <output_dir>/<cell_line>/<plate_id>/
      mmse/mmse_{well}_s{site:02d}_{channel}.tif
      metadata.csv, metrics_summary.csv

Usage:
    python 4_predict.py --manifest .../splits/cpg0036-EU-OS-bioactives/HepG2/split_manifest.csv \\
        --plate_id IMTM_HepG2_B1001_R1 \\
        --training_dir /project/cell_paint_mono/training_datasets/training_dataset_cpg0036_allsites_HepG2 \\
        --output_dir /project/cell_paint_mono/predictions/cpg0036-EU-OS-bioactives --autocast
"""

import argparse
import os
import sys
import time

import torch

from microsplit_reproducibility.workflows.cellpainting import (
    find_checkpoint,
    load_manifest,
    load_model_and_stats,
    manifest_rows_for_plate,
    predict_manifest_fovs,
)

CHANNELS = ["DNA", "ER_RNA", "Actin_AGP", "Mito"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="MicroSplit plate prediction for cpg0036-EU-OS-bioactives")
    p.add_argument("--manifest", required=True, help="split_manifest.csv from 0_split.py")
    p.add_argument("--plate_id", required=True, help="plate_id column value of the plate to predict")
    p.add_argument("--training_dir", required=True,
                   help="Training dataset dir (checkpoints/, noise_models/, training_stats.npz)")
    p.add_argument("--output_dir", required=True,
                   help="Root output dir; {cell_line}/{plate_id} is created below it")
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--mmse_count", type=int, default=50)
    p.add_argument("--batch_size", type=int, default=256, help="tiles per forward pass")
    p.add_argument("--autocast", action="store_true", help="bfloat16 autocast on CUDA")
    p.add_argument("--grid_size", type=int, default=32)
    p.add_argument("--max_fovs", type=int, default=0, help="Process only the first N FOVs (0 = all)")
    p.add_argument("--no_resume", action="store_true", help="Recompute FOVs that already exist")
    p.add_argument("--save_float32", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    torch.set_float32_matmul_precision("high")

    rows = manifest_rows_for_plate(load_manifest(args.manifest), args.plate_id, plate_column="plate_id")
    if rows.empty:
        print(f"ERROR: no test rows for {args.plate_id} in {args.manifest}", file=sys.stderr)
        sys.exit(1)
    if args.max_fovs > 0:
        rows = rows.head(args.max_fovs)
    cell_line = rows.cell_line.iloc[0]
    output_dir = os.path.join(args.output_dir, cell_line, args.plate_id)

    checkpoint = args.checkpoint or find_checkpoint(args.training_dir)
    print(f"Plate:      {args.plate_id} ({cell_line})")
    print(f"Output:     {output_dir}")
    print(f"Checkpoint: {checkpoint}")
    print(f"MMSE:       {args.mmse_count}  tiles/batch: {args.batch_size}  autocast: {args.autocast}")
    model, stats = load_model_and_stats(args.training_dir, checkpoint, channel_names=CHANNELS)

    t0 = time.time()
    predict_manifest_fovs(
        rows, model, stats, CHANNELS, output_dir,
        mmse_count=args.mmse_count, batch_size=args.batch_size, use_autocast=args.autocast,
        grid_size=args.grid_size, save_float32=args.save_float32, resume=not args.no_resume,
    )
    print(f"Done in {(time.time() - t0) / 3600:.2f} h")


if __name__ == "__main__":
    main()
