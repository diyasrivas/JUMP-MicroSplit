#!/usr/bin/env python3
"""
MicroSplit plate-level prediction for cpg0039-garcia-fossa-livecellpainting data.

Live Cell Painting with Acridine Orange — only 2 channels (AO_Green, AO_Red).

For each FOV in a plate:
  1. Reads 2 fluorescence channels from raw TIFF plate images
  2. Creates the combined input (sum of channels)
  3. Tiles with overlapping patches + multi-scale context
  4. Runs model mmse_count times → MMSE prediction + N posterior samples
  5. Saves as uint16 TIFFs
  6. Writes metadata.csv (per-FOV PSNR/SSIM) and metrics_summary.csv

Channel mapping (cpg0039, Cytation 5 BioTek):
  AO_Red   = ch1   (PI filter,  531/647 nm)
  AO_Green = ch2   (GFP filter, 469/525 nm)

  *** Verify channel indices against actual TIFF files before running! ***

Always use the model trained on the same cell line as the plate being predicted.
Four cell lines: Huh_7, MCF_7, PNT1A, PC_3.

Image properties:
  1224 × 904 pixels, 16-bit, 16 sites per well, 96-well plates

Output structure:
  <output_dir>/<plate>/
      mmse/                mmse_{well}_s{site:02d}_{channel}.tif
      sample_0/ ...        sample{N}_{well}_s{site:02d}_{channel}.tif
      metadata.csv
      metrics_summary.csv

Usage:
    python 4_predict.py \\
        --plate_dir  /project/cell_paint_mono/cpg0039-garcia-fossa-livecellpainting/{batch}/images/{plate}/ \\
        --training_dir /project/cell_paint_mono/training_datasets/training_dataset_cpg0039_Huh_7 \\
        --output_dir /project/cell_paint_mono/predictions/cpg0039-garcia-fossa-livecellpainting
"""

import argparse
import gc
import os
import sys
import time

import numpy as np
import pandas as pd
import torch

from microsplit_reproducibility.workflows.cellpainting import (
    discover_fovs,
    read_fov_channels,
    predict_fov,
    save_fov_predictions,
    compute_metrics,
    find_checkpoint,
    load_model_and_stats,
)

# ---------------------------------------------------------------------------
# Dataset-specific configuration
# ---------------------------------------------------------------------------

# 2-channel Live Cell Painting (Acridine Orange)
CHANNELS = ["AO_Green", "AO_Red"]

# cpg0039 channel mapping (Cytation 5 BioTek)
# *** Verify these indices against actual TIFF files! ***
CHANNEL_MAPPING = {
    "AO_Green": 2,   # GFP filter Ex/Em 469/525 nm
    "AO_Red":   1,   # PI filter  Ex/Em 531/647 nm
}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="MicroSplit plate prediction for cpg0039-garcia-fossa-livecellpainting"
    )
    parser.add_argument("--plate_dir",    required=True,
                        help="Plate directory containing Images/")
    parser.add_argument("--training_dir", required=True,
                        help="Training dataset directory for the same cell line "
                             "(contains checkpoints/, noise_models/, training_stats.npz)")
    parser.add_argument("--output_dir",   required=True,
                        help="Root output directory; plate subdirectory created automatically")
    parser.add_argument("--checkpoint",   default=None,
                        help="Path to .ckpt file "
                             "(default: latest best in training_dir/checkpoints/)")
    parser.add_argument("--mmse_count",           type=int, default=50)
    parser.add_argument("--num_posterior_samples", type=int, default=3)
    parser.add_argument("--posterior_seeds", type=int, nargs="+",
                        default=[42, 123, 456])
    parser.add_argument("--batch_size",   type=int, default=32)
    parser.add_argument("--grid_size",    type=int, default=32)
    parser.add_argument("--save_float32", action="store_true",
                        help="Save float32 instead of uint16")
    parser.add_argument("--max_fovs",     type=int, default=0,
                        help="Process only first N FOVs (0 = all)")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    torch.set_float32_matmul_precision("high")

    plate_images_dir = os.path.join(args.plate_dir, "Images")
    if not os.path.isdir(plate_images_dir):
        print(f"ERROR: Images/ not found in {args.plate_dir}", file=sys.stderr)
        sys.exit(1)

    plate_name = os.path.basename(args.plate_dir)
    output_dir = os.path.join(args.output_dir, plate_name)
    os.makedirs(output_dir, exist_ok=True)

    checkpoint = args.checkpoint or find_checkpoint(args.training_dir)
    model, stats = load_model_and_stats(
        training_dir=args.training_dir,
        checkpoint_path=checkpoint,
        channel_names=CHANNELS,
    )

    print(f"Plate:      {plate_name}")
    print(f"Channels:   {CHANNELS}")
    print(f"Checkpoint: {checkpoint}")
    print(f"Output:     {output_dir}")
    print(f"MMSE:       {args.mmse_count}  Posterior samples: {args.num_posterior_samples}")
    print("NOTE: Verify CHANNEL_MAPPING indices against actual TIFF files!")

    fovs = discover_fovs(plate_images_dir)
    total_fovs = len(fovs)
    if args.max_fovs > 0:
        fovs = fovs[:args.max_fovs]
    print(f"FOVs: {total_fovs} found, {len(fovs)} to process")

    plate_meta = {"plate_name": plate_name}

    metadata_rows = []
    t0 = time.time()

    for fov_idx, ((well, site), prefix) in enumerate(fovs):
        fov_t0 = time.time()
        print(f"\n[{fov_idx + 1}/{len(fovs)}] {well} site {site}")

        try:
            channel_images = read_fov_channels(
                plate_images_dir, prefix, CHANNEL_MAPPING
            )
        except Exception as e:
            print(f"  SKIP: {e}")
            continue

        mmse_pred, posterior_samples = predict_fov(
            model=model,
            channel_images=channel_images,
            stats=stats,
            channel_names=CHANNELS,
            image_size=64,
            grid_size=args.grid_size,
            multiscale_lowres_count=3,
            mmse_count=args.mmse_count,
            num_posterior_samples=args.num_posterior_samples,
            posterior_seeds=tuple(args.posterior_seeds[:args.num_posterior_samples]),
            batch_size=args.batch_size,
        )

        save_fov_predictions(
            output_dir=output_dir,
            well=well,
            site=site,
            mmse_prediction=mmse_pred,
            posterior_samples=posterior_samples,
            channel_names=CHANNELS,
            save_uint16=not args.save_float32,
        )

        fov_metrics = compute_metrics(channel_images, mmse_pred, CHANNELS)
        fov_elapsed = time.time() - fov_t0

        row = {**plate_meta}
        row.update({
            "well":            well,
            "site":            site,
            "filename_prefix": prefix,
            "mmse_files": ",".join(
                f"mmse/mmse_{well}_s{site:02d}_{ch}.tif" for ch in CHANNELS
            ),
        })
        for s_idx in range(args.num_posterior_samples):
            row[f"sample_{s_idx}_files"] = ",".join(
                f"sample_{s_idx}/sample{s_idx}_{well}_s{site:02d}_{ch}.tif"
                for ch in CHANNELS
            )
        row.update(fov_metrics)
        metadata_rows.append(row)

        elapsed = time.time() - t0
        rate    = (fov_idx + 1) / elapsed
        eta     = (len(fovs) - fov_idx - 1) / rate if rate > 0 else 0
        psnr_str = " ".join(f"{ch}={fov_metrics[f'psnr_{ch}']:.1f}" for ch in CHANNELS)
        print(
            f"  {fov_elapsed:.1f}s | {fov_idx + 1}/{len(fovs)} | "
            f"ETA {eta / 3600:.1f}h | {psnr_str}"
        )

        if (fov_idx + 1) % 50 == 0:
            torch.cuda.empty_cache()
            gc.collect()

    if metadata_rows:
        meta_df   = pd.DataFrame(metadata_rows)
        meta_path = os.path.join(output_dir, "metadata.csv")
        meta_df.to_csv(meta_path, index=False)
        print(f"\nMetadata → {meta_path}")

        psnr_cols = [f"psnr_{ch}" for ch in CHANNELS]
        ssim_cols = [f"ssim_{ch}" for ch in CHANNELS]
        summary   = [
            {
                "channel":   ch,
                "mean_psnr": meta_df[f"psnr_{ch}"].mean(),
                "mean_ssim": meta_df[f"ssim_{ch}"].mean(),
            }
            for ch in CHANNELS
        ]
        summary.append({
            "channel":   "all_channels",
            "mean_psnr": meta_df[psnr_cols].values.mean(),
            "mean_ssim": meta_df[ssim_cols].values.mean(),
        })
        summary_df   = pd.DataFrame(summary)
        summary_path = os.path.join(output_dir, "metrics_summary.csv")
        summary_df.to_csv(summary_path, index=False)
        print(f"Summary   → {summary_path}")
        print(summary_df.to_string(index=False))

    total_elapsed = time.time() - t0
    print(f"\nComplete: {len(metadata_rows)}/{len(fovs)} FOVs in {total_elapsed / 3600:.2f}h")


if __name__ == "__main__":
    main()
