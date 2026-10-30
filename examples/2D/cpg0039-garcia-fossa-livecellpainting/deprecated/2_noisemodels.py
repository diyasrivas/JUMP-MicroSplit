#!/usr/bin/env python3
"""
Per-channel noise model training for cpg0039-garcia-fossa-livecellpainting MicroSplit.

Only 2 channels — run 2 jobs in parallel via noise_models.sh.

Channels:
  AO_Green — Acridine Orange, GFP filter (469/525 nm), DNA + RNA signal
  AO_Red   — Acridine Orange, PI filter  (531/647 nm), acidic organelle signal

Workflow:
  1. Load all images for one channel from the dataset directory
  2. Train a Noise2Void (N2V) model to obtain denoised predictions
  3. Fit a GaussianMixtureNoiseModel on (signal, prediction) pairs
  4. Save as noise_model_{channel}.npz

Usage:
    python 2_noisemodels.py \\
        --dataset_dir /project/cell_paint_mono/training_datasets/training_dataset_cpg0039_Huh_7 \\
        --output_dir  .../training_dataset_cpg0039_Huh_7/noise_models \\
        --channel AO_Green
"""

import argparse

from microsplit_reproducibility.workflows.noise_model import train_noise_model_for_channel

VALID_CHANNELS = ["AO_Green", "AO_Red"]


def main():
    parser = argparse.ArgumentParser(
        description="Train per-channel noise model for cpg0039-garcia-fossa-livecellpainting"
    )
    parser.add_argument("--dataset_dir", required=True,
                        help="Path to training_dataset_cpg0039_{cell_line}/")
    parser.add_argument("--output_dir",  required=True,
                        help="Directory to write noise_model_{channel}.npz")
    parser.add_argument("--channel",     required=True,
                        choices=VALID_CHANNELS,
                        help="Channel to process")
    args = parser.parse_args()

    train_noise_model_for_channel(
        channel=args.channel,
        dataset_dir=args.dataset_dir,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
