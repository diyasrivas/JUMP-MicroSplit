#!/usr/bin/env python3
"""
Per-channel noise model training for cpg0036-EU-OS-bioactives MicroSplit.

Trains one job per channel; run 4 jobs in parallel via noise_models.sh.

Channels (4 total):
  DNA       — Hoechst 33342          (435-480 nm)
  ER_RNA    — ConA Alexa 488 + SYTO14 (500-550 nm, spectrally mixed)
  Actin_AGP — WGA Alexa 555 + Phalloidin Alexa 568 (570-630 nm, spectrally mixed)
  Mito      — MitoTracker Deep Red   (650-760 nm)

Workflow:
  1. Load all images for one channel from the dataset directory
  2. Train a Noise2Void (N2V) model to obtain denoised predictions
  3. Fit a GaussianMixtureNoiseModel on (signal, prediction) pairs
  4. Save as noise_model_{channel}.npz

Usage:
    python 2_noisemodels.py \\
        --dataset_dir /project/cell_paint_mono/training_datasets/training_dataset_cpg0036_HepG2 \\
        --output_dir  .../training_dataset_cpg0036_HepG2/noise_models \\
        --channel DNA
"""

import argparse

from microsplit_reproducibility.workflows.noise_model import train_noise_model_for_channel

VALID_CHANNELS = ["DNA", "ER_RNA", "Actin_AGP", "Mito"]


def main():
    parser = argparse.ArgumentParser(
        description="Train per-channel noise model for cpg0036-EU-OS-bioactives"
    )
    parser.add_argument("--dataset_dir", required=True,
                        help="Path to training_dataset_cpg0036_{cell_line}/")
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
