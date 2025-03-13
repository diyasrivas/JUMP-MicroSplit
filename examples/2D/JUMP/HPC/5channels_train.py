#!/usr/bin/env python3
import os
import sys
import numpy as np
import tifffile
from pathlib import Path

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
    
    Parameters
    ----------
    dataset_dir : str
        Path to the dataset directory
    channel_list : list
        List of channel names to verify
        
    Returns
    -------
    bool
        True if all required directories and files exist, False otherwise
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
    
    Parameters
    ----------
    noise_models_dir : str
        Path to the noise models directory
    channel_list : list
        List of channel names to verify
        
    Returns
    -------
    bool
        True if all required noise models exist, False otherwise
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


class JUMPDataset(Dataset):
    """
    Custom Dataset class for JUMP Cell Painting data that supports flexible channel selection.
    This class loads TIFF files from the prepared dataset directory structure and combines
    them as required for μSplit training.
    """
    def __init__(self, data_dir, channel_list, transform=None):
        """
        Initialize the dataset.
        
        Parameters
        ----------
        data_dir : str
            Path to the dataset directory
        channel_list : list
            List of channel names to load
        transform : callable, optional
            Optional transform to be applied on a sample
        """
        self.data_dir = data_dir
        self.channel_list = channel_list
        self.transform = transform
        
        # Verify dataset structure
        assert verify_dataset_structure(data_dir, channel_list), "Dataset structure is invalid"
        
        # Get list of files for each channel
        self.files = {}
        for channel in channel_list + ["combined"]:
            channel_dir = os.path.join(data_dir, channel)
            self.files[channel] = sorted([f for f in os.listdir(channel_dir) if f.endswith('.tif')])
            
        # Check that all channels have the same number of files
        file_counts = [len(files) for files in self.files.values()]
        assert len(set(file_counts)) == 1, f"Channel directories have different file counts: {file_counts}"
        
        # Load data
        self._load_data()
        
    def _load_data(self):
        """Load all data into memory."""
        # Load individual channel images
        channel_images = []
        for channel in self.channel_list:
            channel_dir = os.path.join(self.data_dir, channel)
            images = []
            for file in self.files[channel]:
                img = tifffile.imread(os.path.join(channel_dir, file))
                images.append(img)
            channel_images.append(np.stack(images))
            
        # Load combined channel images
        combined_dir = os.path.join(self.data_dir, "combined")
        combined_images = []
        for file in self.files["combined"]:
            img = tifffile.imread(os.path.join(combined_dir, file))
            combined_images.append(img)
        combined_images = np.stack(combined_images)
        
        channel_stack = np.stack(channel_images, axis=-1)
        self._data = np.concatenate([channel_stack, combined_images[..., np.newaxis]], axis=-1)
    
    def __len__(self):
        """Return the number of samples in the dataset."""
        return len(self.files[self.channel_list[0]])
    
    def __getitem__(self, idx):
        """
        Get a sample from the dataset.
        
        Parameters
        ----------
        idx : int
            Index of the sample to get
            
        Returns
        -------
        tuple
            (input, target) where input is the combined channel and target is the individual channels
        """
        # Get data for this index
        sample = self._data[idx]
        
        # Split into individual channels and combined channel
        individual_channels = sample[..., :-1]
        combined_channel = sample[..., -1]
        
        # Apply transform if specified
        if self.transform:
            individual_channels = self.transform(individual_channels)
            combined_channel = self.transform(combined_channel)
            
        return combined_channel, individual_channels
    
    def get_num_frames(self):
        """Return the number of frames in the dataset."""
        return len(self)
    
    def reduce_data(self, indices):
        """
        Reduce the dataset to a subset of frames.
        
        Parameters
        ----------
        indices : list
            List of indices to keep
        """
        self._data = self._data[indices]
        for channel in self.channel_list + ["combined"]:
            self.files[channel] = [self.files[channel][i] for i in indices]
        print(f"[JUMPDataset] Data reduced. New data shape: {self._data.shape}")


def main():
    # HARDCODED PARAMETERS FOR 5-CHANNEL DATASET
    TARGET_CHANNEL_LIST = ["DNA", "RNA", "ER", "AGP", "Mito"]
    num_epochs = 10
    batch_size = 8
    reduce_data = False
    
    # Print configuration
    print(f"Selected channels: {TARGET_CHANNEL_LIST}")
    print(f"Training epochs: {num_epochs}")
    print(f"Batch size: {batch_size}")
    
    # Set up paths
    channel_dir_name = "_".join([ch.lower() for ch in TARGET_CHANNEL_LIST])
    num_channels = len(TARGET_CHANNEL_LIST)
    DATASET_DIR = f"experiments/{num_channels}_channels/{channel_dir_name}"
    NOISE_MODELS_DIR = "noise_models"
    
    print(f"Dataset directory: {DATASET_DIR}")
    print(f"Noise models directory: {NOISE_MODELS_DIR}")
    
    # Ensure all required data and noise models exist
    assert os.path.exists(DATASET_DIR), f"Dataset directory '{DATASET_DIR}' does not exist"
    
    for channel in TARGET_CHANNEL_LIST:
        channel_dir = os.path.join(DATASET_DIR, channel)
        assert os.path.exists(channel_dir), f"Channel directory '{channel_dir}' does not exist"
    
    combined_dir = os.path.join(DATASET_DIR, "combined")
    assert os.path.exists(combined_dir), f"Combined channel directory '{combined_dir}' does not exist"
    
    for channel in TARGET_CHANNEL_LIST:
        noise_model_path = os.path.join(NOISE_MODELS_DIR, f"noise_model_{channel}.npz")
        assert os.path.exists(noise_model_path), f"Noise model '{noise_model_path}' does not exist"
    
    print("✓ All required data and noise models are available!")
    
    # Setting up data configs
    train_data_config, val_data_config, test_data_config = get_data_configs(
        channel_idx_list=TARGET_CHANNEL_LIST,
    )
    
    # Setting up MicroSplit parameters
    experiment_params = get_microsplit_parameters(
        nm_path=NOISE_MODELS_DIR,
        channel_idx_list=TARGET_CHANNEL_LIST,
        batch_size=batch_size
    )
    
    # Create datasets
    train_dset, val_dset, test_dset, data_stats = create_train_val_datasets(
        datapath=DATASET_DIR,
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

if __name__ == "__main__":
    main()
