#!/usr/bin/env python3
"""
MicroSplit training for cpg0000-jump-pilot Cell Painting dataset.

Expected dataset layout under <dataset_dir>/:
    DNA/ RNA/ ER/ AGP/ Mito/   — individual channel .tiff files (uint16)
    combined/                   — pixel-wise sum of all channels (float32)
    noise_models/               — noise_model_{channel}.npz per channel
    checkpoints/                — written here during training

Usage:
    python 3_train.py --dataset_dir /project/cell_paint_mono/training_datasets/training_dataset_A549
    python 3_train.py --dataset_dir ... --test_mode   # 2 epochs sanity check
"""

import argparse
import glob
import os
import sys

import numpy as np
import torch
from pytorch_lightning import Trainer
from torch.utils.data import DataLoader

from careamics.lightning import VAEModule

from microsplit_reproducibility.configs.data.JUMP import get_data_configs
from microsplit_reproducibility.configs.factory import (
    create_algorithm_config,
    get_likelihood_config,
    get_loss_config,
    get_model_config,
    get_lr_scheduler_config,
    get_optimizer_config,
    get_training_config,
)
from microsplit_reproducibility.configs.parameters.JUMP import get_microsplit_parameters
from microsplit_reproducibility.datasets.common import create_lazy_datasets
from microsplit_reproducibility.utils.callbacks import get_callbacks


CHANNELS = ["DNA", "RNA", "ER", "AGP", "Mito"]


def parse_args():
    parser = argparse.ArgumentParser(
        description="MicroSplit training for cpg0000-jump-pilot"
    )
    parser.add_argument("--dataset_dir",  required=True,
                        help="Path to the training dataset directory")
    parser.add_argument("--num_epochs",   type=int,   default=100)
    parser.add_argument("--batch_size",   type=int,   default=64)
    parser.add_argument("--num_workers",  type=int,   default=8)
    parser.add_argument("--check_val_every_n_epoch", type=int, default=5)
    parser.add_argument("--early_stopping_patience", type=int, default=50)
    parser.add_argument("--test_mode",    action="store_true",
                        help="Run 2 epochs on 2 frames for a sanity check")
    parser.add_argument("--compile",      action="store_true",
                        help="torch.compile the inner LadderVAE for faster training")
    parser.add_argument("--train_grid_size", type=int, default=64,
                        help="Training grid size — larger = fewer patches per epoch")
    parser.add_argument("--val_fraction",  type=float, default=0.1)
    parser.add_argument("--test_fraction", type=float, default=0.1)
    parser.add_argument("--cache_size",    type=int,   default=64,
                        help="LRU cache size in frames")
    return parser.parse_args()


def validate_inputs(dataset_dir: str, noise_models_dir: str):
    missing = []
    for ch in CHANNELS + ["combined"]:
        d = os.path.join(dataset_dir, ch)
        if not os.path.isdir(d):
            missing.append(f"Missing directory: {d}")
        elif not glob.glob(os.path.join(d, "*.tif*")):
            missing.append(f"No TIFF files in: {d}")
    for ch in CHANNELS:
        nm = os.path.join(noise_models_dir, f"noise_model_{ch}.npz")
        if not os.path.isfile(nm):
            missing.append(f"Missing noise model: {nm}")
    if missing:
        for m in missing:
            print(f"ERROR: {m}", file=sys.stderr)
        sys.exit(1)


def main():
    args = parse_args()
    torch.set_float32_matmul_precision("high")

    dataset_dir      = args.dataset_dir
    noise_models_dir = os.path.join(dataset_dir, "noise_models")
    checkpoint_dir   = os.path.join(dataset_dir, "checkpoints")

    validate_inputs(dataset_dir, noise_models_dir)

    num_epochs = 2 if args.test_mode else args.num_epochs
    check_val  = 1 if args.test_mode else args.check_val_every_n_epoch

    print(f"Dataset:         {dataset_dir}")
    print(f"Noise models:    {noise_models_dir}")
    print(f"Channels:        {CHANNELS}")
    print(f"Epochs:          {num_epochs}")
    print(f"Batch size:      {args.batch_size}")
    print(f"Train grid size: {args.train_grid_size}")
    print(f"Test mode:       {args.test_mode}")

    train_data_config, val_data_config, _ = get_data_configs(
        channel_idx_list=CHANNELS,
        train_grid_size=args.train_grid_size,
    )

    experiment_params = get_microsplit_parameters(
        nm_path=noise_models_dir,
        channel_idx_list=CHANNELS,
        batch_size=args.batch_size,
    )

    train_dset, val_dset, test_dset, data_stats = create_lazy_datasets(
        datapath=dataset_dir,
        channel_names=CHANNELS,
        train_grid_size=args.train_grid_size,
        val_grid_size=32,
        image_size=64,
        multiscale_lowres_count=train_data_config.multiscale_lowres_count,
        val_fraction=args.val_fraction,
        test_fraction=args.test_fraction,
        cache_size=args.cache_size,
        enable_rotation=train_data_config.train_aug_rotate,
    )

    # Save training stats for the prediction script
    mean_dict, std_dict = train_dset.get_mean_std()
    stats_path = os.path.join(dataset_dir, "training_stats.npz")
    np.savez(
        stats_path,
        mean_input  = np.array(mean_dict["input"]),
        std_input   = np.array(std_dict["input"]),
        mean_target = np.array(mean_dict["target"]),
        std_target  = np.array(std_dict["target"]),
        max_val     = np.array(train_dset.get_max_val()),
    )
    print(f"Training stats saved: {stats_path}")

    if args.test_mode:
        train_dset.reduce_data([0, 1])
        val_dset.reduce_data([0])

    print(f"Train frames: {train_dset.get_num_frames()}")
    print(f"Val frames:   {val_dset.get_num_frames()}")

    loader_kwargs = dict(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
        prefetch_factor=2 if args.num_workers > 0 else None,
    )
    train_loader = DataLoader(train_dset, shuffle=True,  **loader_kwargs)
    val_loader   = DataLoader(val_dset,   shuffle=False, **loader_kwargs)

    experiment_params["data_stats"] = data_stats

    loss_config          = get_loss_config(**experiment_params)
    model_config         = get_model_config(**experiment_params)
    gaussian_lik_config, noise_model_config, nm_lik_config = \
        get_likelihood_config(**experiment_params)
    training_config      = get_training_config(**experiment_params)
    lr_scheduler_config  = get_lr_scheduler_config(**experiment_params)
    optimizer_config     = get_optimizer_config(**experiment_params)

    training_config.num_epochs = num_epochs

    experiment_config = create_algorithm_config(
        algorithm=experiment_params["algorithm"],
        loss_config=loss_config,
        model_config=model_config,
        gaussian_lik_config=gaussian_lik_config,
        nm_config=noise_model_config,
        nm_lik_config=nm_lik_config,
        lr_scheduler_config=lr_scheduler_config,
        optimizer_config=optimizer_config,
    )

    model = VAEModule(algorithm_config=experiment_config)

    if args.compile:
        # Compile only the inner LadderVAE (not VAEModule) to avoid dynamo
        # graph breaks in validation_step.
        model.model = torch.compile(model.model)

    os.makedirs(checkpoint_dir, exist_ok=True)
    callbacks = get_callbacks(checkpoint_dir)
    for cb in callbacks:
        if hasattr(cb, "patience"):
            cb.patience = args.early_stopping_patience

    trainer = Trainer(
        max_epochs=num_epochs,
        accelerator="gpu",
        enable_progress_bar=True,
        callbacks=callbacks,
        precision=training_config.precision,
        gradient_clip_val=training_config.gradient_clip_val,
        gradient_clip_algorithm=training_config.gradient_clip_algorithm,
        check_val_every_n_epoch=check_val,
        default_root_dir=dataset_dir,
    )

    trainer.fit(
        model=model,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader,
    )

    print(f"Training complete. Checkpoints: {checkpoint_dir}")


if __name__ == "__main__":
    main()
