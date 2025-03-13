#!/usr/bin/env python3
import os
import sys
import numpy as np
import tifffile
from pathlib import Path
import argparse
import itertools

# Import deep learning frameworks
import torch
from torch.utils.data import DataLoader, Dataset
from pytorch_lightning import Trainer

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


def get_target(dataset):
    """Extract target channels from the dataset."""
    return dataset._data[..., :-1]


def get_input(dataset):
    """Extract input (combined) channel from the dataset."""
    return dataset._data[..., -1]


def get_3channel_combinations():
    """Get all 3-channel combinations from 5 available channels"""
    ALL_CHANNELS = ["DNA", "RNA", "ER", "AGP", "Mito"]
    return list(itertools.combinations(ALL_CHANNELS, 3))


def find_existing_3channel_dirs():
    """Find existing 3-channel directories in experiments/3_channels"""
    base_dir = "experiments/3_channels"
    
    if not os.path.exists(base_dir):
        print(f"Directory {base_dir} does not exist")
        return []
    
    return [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]


def get_channel_list_from_dir_name(dir_name):
    """Get channel list from directory name"""
    parts = dir_name.split('_')
    # Convert to proper case for channels
    channel_map = {'dna': 'DNA', 'rna': 'RNA', 'er': 'ER', 'agp': 'AGP', 'mito': 'Mito'}
    return [channel_map.get(part, part) for part in parts]


def train_model(target_channel_list, num_epochs=10, batch_size=16, reduce_data=False):
    """Train a model for the specified channel combination"""
    # Print configuration
    print(f"Selected channels: {target_channel_list}")
    print(f"Training epochs: {num_epochs}")
    print(f"Batch size: {batch_size}")
    
    # Set up paths
    channel_dir_name = "_".join([ch.lower() for ch in target_channel_list])
    num_channels = len(target_channel_list)
    dataset_dir = f"experiments/{num_channels}_channels/{channel_dir_name}"
    noise_models_dir = "noise_models"
    
    print(f"Dataset directory: {dataset_dir}")
    print(f"Noise models directory: {noise_models_dir}")
    
    # Ensure all required data and noise models exist
    assert os.path.exists(dataset_dir), f"Dataset directory '{dataset_dir}' does not exist"
    
    for channel in target_channel_list:
        channel_dir = os.path.join(dataset_dir, channel)
        assert os.path.exists(channel_dir), f"Channel directory '{channel_dir}' does not exist"
    
    combined_dir = os.path.join(dataset_dir, "combined")
    assert os.path.exists(combined_dir), f"Combined channel directory '{combined_dir}' does not exist"
    
    for channel in target_channel_list:
        noise_model_path = os.path.join(noise_models_dir, f"noise_model_{channel}.npz")
        assert os.path.exists(noise_model_path), f"Noise model '{noise_model_path}' does not exist"
    
    print("✓ All required data and noise models are available!")
    
    # Setting up data configs
    train_data_config, val_data_config, test_data_config = get_data_configs(
        channel_idx_list=target_channel_list,
    )
    
    # Setting up MicroSplit parameters
    experiment_params = get_microsplit_parameters(
        nm_path=noise_models_dir,
        channel_idx_list=target_channel_list,
        batch_size=batch_size
    )
    
    # Create datasets
    train_dset, val_dset, test_dset, data_stats = create_train_val_datasets(
        datapath=dataset_dir,
        train_config=train_data_config,
        val_config=val_data_config,
        test_config=test_data_config,
        load_data_func=get_train_val_data,
    )
    
    print(f"Training dataset: {train_dset.get_num_frames()} frames")
    print(f"Validation dataset: {val_dset.get_num_frames()} frames")
    print(f"Test dataset: {test_dset.get_num_frames()} frames")
    
    # Optionally reduce dataset size for testing
    if reduce_data:
        print("Using REDUCED training and validation data for quick testing!")
        train_dset.reduce_data([0, 1])
        val_dset.reduce_data([0])
        print(f"Reduced training dataset: {train_dset.get_num_frames()} frames")
        print(f"Reduced validation dataset: {val_dset.get_num_frames()} frames")
    else:
        print('Using the full set of training and validation data!')
        print(f"(This includes {train_dset.get_num_frames()} and {val_dset.get_num_frames()} frames, respectively.)")
    
    # Create data loaders
    train_dloader = DataLoader(
        train_dset,
        batch_size=experiment_params["batch_size"],
        num_workers=experiment_params.get("num_workers", 4),
        shuffle=True,
    )
    
    val_dloader = DataLoader(
        val_dset,
        batch_size=experiment_params["batch_size"],
        num_workers=experiment_params.get("num_workers", 4),
        shuffle=False,
    )
    
    # Model configuration
    experiment_params["data_stats"] = data_stats
    
    loss_config = get_loss_config(**experiment_params)
    model_config = get_model_config(**experiment_params)
    gaussian_lik_config, noise_model_config, nm_lik_config = get_likelihood_config(
        **experiment_params
    )
    
    training_config = get_training_config(**experiment_params)
    training_config.num_epochs = num_epochs
    
    lr_scheduler_config = get_lr_scheduler_config(**experiment_params)
    optimizer_config = get_optimizer_config(**experiment_params)
    
    # Final configuration
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
    
    # Initialize model
    model = VAEModule(algorithm_config=experiment_config)
    print("Model initialized successfully")
    
    # Create checkpoint directory
    checkpoint_dir = f"./checkpoints/{num_channels}_channels/{channel_dir_name}"
    os.makedirs(checkpoint_dir, exist_ok=True)
    print(f"Checkpoint directory: {checkpoint_dir}")
    
    # Create trainer
    trainer = Trainer(
        max_epochs=training_config.num_epochs,
        accelerator="gpu",
        enable_progress_bar=True,
        callbacks=get_callbacks(checkpoint_dir),
        precision=training_config.precision,
        gradient_clip_val=training_config.gradient_clip_val,
        gradient_clip_algorithm=training_config.gradient_clip_algorithm,
    )
    
    # Train model
    trainer.fit(
        model=model,
        train_dataloaders=train_dloader,
        val_dataloaders=val_dloader,
    )
    
    print("Training complete!")
    print(f"Model checkpoints saved to: {checkpoint_dir}")


def main():
    parser = argparse.ArgumentParser(description='Train MicroSplit model for 3-channel combinations')
    parser.add_argument('--index', type=int, required=True, help='Index of the 3-channel combination (0-9)')
    parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=16, help='Batch size for training')
    parser.add_argument('--reduce-data', action='store_true', help='Reduce dataset size for testing')
    
    # Try to get SLURM array task ID if not explicitly provided
    if len(sys.argv) == 1:
        slurm_array_task_id = os.environ.get('SLURM_ARRAY_TASK_ID')
        if slurm_array_task_id is not None:
            print(f"Using SLURM_ARRAY_TASK_ID: {slurm_array_task_id}")
            sys.argv.extend(['--index', slurm_array_task_id])
    
    args = parser.parse_args()
    
    # Find existing 3-channel directories
    existing_dirs = find_existing_3channel_dirs()
    if not existing_dirs:
        print("No 3-channel directories found. Please create the datasets first.")
        sys.exit(1)
    
    sorted_dirs = sorted(existing_dirs)
    
    if args.index < 0 or args.index >= len(sorted_dirs):
        print(f"Invalid index: {args.index}. Must be between 0 and {len(sorted_dirs)-1}")
        print("Available directories:")
        for i, dir_name in enumerate(sorted_dirs):
            print(f"  {i}: {dir_name}")
        sys.exit(1)
    
    dir_name = sorted_dirs[args.index]
    channel_list = get_channel_list_from_dir_name(dir_name)
    
    print(f"Selected directory: {dir_name}")
    print(f"Corresponding channels: {channel_list}")
    
    # Train the model
    train_model(
        target_channel_list=channel_list,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        reduce_data=args.reduce_data
    )


if __name__ == "__main__":
    main()

