#!/usr/bin/env python3
"""
Script to train a MicroSplit model on JUMP dataset with specified channels
Modified from original 02JUMP_train.ipynb notebook to support command-line arguments
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tifffile
from pathlib import Path

# Import deep learning frameworks
import torch
from torch.utils.data import DataLoader, Dataset
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint

# Import CAREamics components
from careamics.lightning import VAEModule
from careamics.lvae_training.dataset import DataSplitType

# Import microsplit reproducibility components
from microsplit_reproducibility.configs.factory import (
    create_algorithm_config,
    get_likelihood_config,
    get_loss_config,
    get_model_config,
    get_optimizer_config,
    get_training_config,
    get_lr_scheduler_config,
)
from microsplit_reproducibility.utils.callbacks import get_callbacks
from microsplit_reproducibility.utils.io import load_checkpoint, load_checkpoint_path
from microsplit_reproducibility.datasets import create_train_val_datasets
from microsplit_reproducibility.utils.utils import (
    plot_training_metrics,
    plot_input_patches,
    plot_training_outputs,
)

# Dataset specific modules
from microsplit_reproducibility.configs.parameters.JUMP import get_microsplit_parameters
from microsplit_reproducibility.configs.data.JUMP import get_data_configs
from microsplit_reproducibility.datasets.JUMP import get_train_val_data


def verify_dataset_structure(dataset_dir, channel_list):
    """
    Verify that the dataset directory has the expected structure with all required channels.
    """
    # Check that the main directory exists
    if not os.path.exists(dataset_dir):
        print(f"ERROR: Dataset directory '{dataset_dir}' does not exist!")
        return False
    
    # Check that all channel directories exist
    for channel in channel_list:
        channel_dir = os.path.join(dataset_dir, channel)
        if not os.path.exists(channel_dir):
            print(f"ERROR: Channel directory '{channel_dir}' does not exist!")
            return False
    
    # Check that the combined directory exists
    combined_dir = os.path.join(dataset_dir, "combined")
    if not os.path.exists(combined_dir):
        print(f"ERROR: Combined channel directory '{combined_dir}' does not exist!")
        return False
    
    # Count files in each directory to ensure consistency
    file_counts = {}
    for channel in channel_list + ["combined"]:
        channel_dir = os.path.join(dataset_dir, channel)
        files = [f for f in os.listdir(channel_dir) if f.endswith('.tif')]
        file_counts[channel] = len(files)
    
    # Check that all directories have the same number of files
    if len(set(file_counts.values())) != 1:
        print(f"WARNING: Channel directories have different file counts: {file_counts}")
        return False
    
    return True


def verify_noise_models(noise_models_dir, channel_list):
    """
    Verify that noise models exist for all channels.
    """
    if not os.path.exists(noise_models_dir):
        print(f"ERROR: Noise models directory '{noise_models_dir}' does not exist!")
        return False
    
    for channel in channel_list:
        noise_model_path = os.path.join(noise_models_dir, f"noise_model_{channel}.npz")
        if not os.path.exists(noise_model_path):
            print(f"ERROR: Noise model '{noise_model_path}' does not exist!")
            return False
    
    return True


def main(args):
    """Main function to run MicroSplit training"""
    print(f"\n{'='*80}")
    print(f"Training MicroSplit model for channels: {args.channels}")
    print(f"{'='*80}")
    
    # Parse channels
    TARGET_CHANNEL_LIST = args.channels.split(',')
    print(f"Selected channels: {TARGET_CHANNEL_LIST}")
    
    # Create a dataset-specific name
    dataset_name = '_'.join([ch.lower() for ch in TARGET_CHANNEL_LIST])
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Redirect stdout to both console and log file if requested
    log_file = os.path.join(args.output_dir, "training_log.txt")
    if args.log_to_file:
        sys.stdout = open(log_file, 'w')
    
    # Check data and noise models
    if not verify_dataset_structure(args.dataset_dir, TARGET_CHANNEL_LIST):
        print(f"Dataset verification failed for {args.dataset_dir}")
        return 1
    
    if not verify_noise_models(args.noise_models_dir, TARGET_CHANNEL_LIST):
        print(f"Noise model verification failed for {args.noise_models_dir}")
        return 1
    
    # Setting up data configs
    train_data_config, val_data_config, test_data_config = get_data_configs(
        channel_idx_list=TARGET_CHANNEL_LIST,
    )
    
    # Setting up MicroSplit parameters
    experiment_params = get_microsplit_parameters(
        nm_path=args.noise_models_dir,
        channel_idx_list=TARGET_CHANNEL_LIST,
        batch_size=args.batch_size
    )
    
    # Create datasets for training, validation, and testing
    train_dset, val_dset, test_dset, data_stats = create_train_val_datasets(
        datapath=args.dataset_dir,
        train_config=train_data_config,
        val_config=val_data_config,
        test_config=test_data_config,
        load_data_func=get_train_val_data,
    )
    
    print(f"Training dataset: {train_dset.get_num_frames()} frames")
    print(f"Validation dataset: {val_dset.get_num_frames()} frames")
    print(f"Test dataset: {test_dset.get_num_frames()} frames")
    
    # Optionally reduce data for testing
    if args.reduce_data:
        print("Using REDUCED training and validation data for quick testing!")
        train_dset.reduce_data([0, 1])
        val_dset.reduce_data([0])
        print(f"Reduced training dataset: {train_dset.get_num_frames()} frames")
        print(f"Reduced validation dataset: {val_dset.get_num_frames()} frames")
    
    # Create data loaders
    train_dloader = DataLoader(
        train_dset,
        batch_size=experiment_params["batch_size"],
        num_workers=args.num_workers,
        shuffle=True,
    )
    
    val_dloader = DataLoader(
        val_dset,
        batch_size=experiment_params["batch_size"],
        num_workers=args.num_workers,
        shuffle=False,
    )
    
    # Make our data_stats known to the experiment
    experiment_params["data_stats"] = data_stats
    
    # Setting up training config
    loss_config = get_loss_config(**experiment_params)
    model_config = get_model_config(**experiment_params)
    gaussian_lik_config, noise_model_config, nm_lik_config = get_likelihood_config(
        **experiment_params
    )
    training_config = get_training_config(**experiment_params)
    training_config.num_epochs = args.epochs
    
    # Setting up learning rate scheduler and optimizer
    lr_scheduler_config = get_lr_scheduler_config(**experiment_params)
    optimizer_config = get_optimizer_config(**experiment_params)
    
    # Final config
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
    
    print(f'Will train for {training_config.num_epochs} epochs')
    
    # Initialize the model
    model = VAEModule(algorithm_config=experiment_config)
    print("Model initialized successfully")
    
    # Create a checkpoint directory for this specific run
    checkpoint_dir = os.path.join(args.output_dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Create custom callbacks
    # This ensures each training job has its own checkpoint directory
    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename=f"{dataset_name}-{{epoch}}-{{val_loss:.2f}}",
        save_top_k=3,
        monitor="val_loss",
        mode="min",
        save_last=True,
    )
    
    # Create a Trainer
    trainer = Trainer(
        max_epochs=training_config.num_epochs,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        enable_progress_bar=True,
        callbacks=[checkpoint_callback],  # Use our custom callback
        precision=training_config.precision,
        gradient_clip_val=training_config.gradient_clip_val,
        gradient_clip_algorithm=training_config.gradient_clip_algorithm,
    )
    
    # Start the training
    trainer.fit(
        model=model,
        train_dataloaders=train_dloader,
        val_dataloaders=val_dloader,
    )
    
    print(f"Training complete! Model saved to {checkpoint_dir}")
    print(f"Best model path: {checkpoint_callback.best_model_path}")
    
    return 0


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Train MicroSplit model on JUMP dataset')
    
    # Required arguments
    parser.add_argument('--channels', type=str, required=True,
                        help='Comma-separated list of channels to use (e.g., "DNA,Mito")')
    
    # Optional arguments with defaults
    parser.add_argument('--dataset_dir', type=str, default='experiments',
                        help='Directory containing the dataset')
    parser.add_argument('--noise_models_dir', type=str, default='noise_models',
                        help='Directory containing the noise models')
    parser.add_argument('--output_dir', type=str, default='training_results',
                        help='Directory to save the training results')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Batch size for training')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data loader workers')
    parser.add_argument('--reduce_data', action='store_true',
                        help='Reduce dataset size for quick testing')
    parser.add_argument('--log_to_file', action='store_true',
                        help='Log output to a file in the output directory')
    parser.add_argument('--task_id', type=int, default=-1,
                        help='Task ID for array jobs')
    
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    sys.exit(main(args))
