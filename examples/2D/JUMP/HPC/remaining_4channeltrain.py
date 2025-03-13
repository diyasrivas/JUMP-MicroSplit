# Import standard libraries
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tifffile
from pathlib import Path

# Import deep learning frameworks
import torch
from torch.utils.data import DataLoader
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

# Base directory - use absolute path
BASE_DIR = "/home/diya.srivastava/Desktop/repos/JUMP-MicroSplit/examples/2D/JUMP"

# Define the channels for your dataset
class Channels:
    DNA = "DNA"
    Mito = "Mito"
    RNA = "RNA"
    ER = "ER"
    AGP = "AGP"

# Select the four channels you want to unmix
TARGET_CHANNEL_LIST = [Channels.DNA, Channels.RNA, Channels.ER, Channels.AGP]
print(f"Selected channels: {TARGET_CHANNEL_LIST}")

# Find directory where dataset is stored - use absolute paths
channel_dir_name = "_".join(ch.lower() for ch in TARGET_CHANNEL_LIST)
num_channels = len(TARGET_CHANNEL_LIST)
DATASET_DIR = os.path.join(BASE_DIR, "experiments", f"{num_channels}_channels", channel_dir_name)
NOISE_MODELS_DIR = os.path.join(BASE_DIR, "noise_models")

print(f"Dataset directory: {DATASET_DIR}")
print(f"Noise models directory: {NOISE_MODELS_DIR}")

# Check if dataset exists
if not os.path.exists(DATASET_DIR):
    raise ValueError(f"Dataset directory '{DATASET_DIR}' does not exist")

# Check if combined directory exists
combined_dir = os.path.join(DATASET_DIR, "combined")
if not os.path.exists(combined_dir):
    raise ValueError(f"Combined channel directory '{combined_dir}' does not exist")

# Check if all channel directories exist
for channel in TARGET_CHANNEL_LIST:
    channel_dir = os.path.join(DATASET_DIR, channel)
    if not os.path.exists(channel_dir):
        raise ValueError(f"Channel directory '{channel_dir}' does not exist")

# Check if noise models exist
for channel in TARGET_CHANNEL_LIST:
    noise_model_path = os.path.join(NOISE_MODELS_DIR, f"noise_model_{channel}.npz")
    if not os.path.exists(noise_model_path):
        raise ValueError(f"Noise model '{noise_model_path}' does not exist. Please run the 01JUMP_noisemodels notebook first.")

print("✓ All required data and noise models are available!")

# Get data configurations
train_data_config, val_data_config, test_data_config = get_data_configs(
    channel_idx_list=TARGET_CHANNEL_LIST,
)

# Setting up MicroSplit parameters
experiment_params = get_microsplit_parameters(
    nm_path=NOISE_MODELS_DIR,
    channel_idx_list=TARGET_CHANNEL_LIST,
    batch_size=16
)

# Create datasets for training, validation, and testing
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

# Optional: Reduce dataset size for quick testing
reduce_data = False
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

# Make our data_stats known to the experiment
experiment_params["data_stats"] = data_stats

# Setting up training losses and model config
loss_config = get_loss_config(**experiment_params)
model_config = get_model_config(**experiment_params)
gaussian_lik_config, noise_model_config, nm_lik_config = get_likelihood_config(
    **experiment_params
)
training_config = get_training_config(**experiment_params)

# Set number of training epochs
num_epochs = 10  # You can adjust this as needed
training_config.num_epochs = num_epochs
print(f'Will train for {training_config.num_epochs} epochs')

# Setting up learning rate scheduler and optimizer
lr_scheduler_config = get_lr_scheduler_config(**experiment_params)
optimizer_config = get_optimizer_config(**experiment_params)

# Create algorithm config
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

# Initialize the model
model = VAEModule(algorithm_config=experiment_config)
print("Model initialized successfully")

# Show some training data for inspection
sample_indices = plot_input_patches(
    dataset=train_dset, 
    num_channels=len(TARGET_CHANNEL_LIST), 
    num_samples=3, 
    patch_size=128
)
print(f"Sample indices: {sample_indices}")

# Create checkpoint directory - absolute path
checkpoint_dir = os.path.join(BASE_DIR, "checkpoints", f"{num_channels}_channels", channel_dir_name)
os.makedirs(checkpoint_dir, exist_ok=True)
print(f"Checkpoint directory: {checkpoint_dir}")

# Create a trainer
trainer = Trainer(
    max_epochs=training_config.num_epochs,
    accelerator="gpu",  # Change to "cpu" if no GPU is available
    enable_progress_bar=True,
    callbacks=get_callbacks(checkpoint_dir),
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

print("Training complete!")

# Look at the training loss curves
from pandas import read_csv

def find_recent_metrics():
    import glob
    # Use absolute path for finding metrics
    metrics_path = os.path.join(BASE_DIR, "lightning_logs", "version_*")
    log_dirs = sorted(glob.glob(metrics_path))
    if log_dirs:
        return os.path.join(log_dirs[-1], "metrics.csv")
    return None

# Plot metrics
metrics_file = find_recent_metrics()
if metrics_file:
    df = read_csv(metrics_file)
    # Use a simplified version for plotting
    fig, ax = plt.subplots(figsize=(12,3), ncols=4)
    
    if 'reconstruction_loss_epoch' in df.columns:
        df['reconstruction_loss_epoch'].dropna().reset_index(drop=True).plot(ax=ax[0], marker='o')
        ax[0].set_title('Reconstruction Loss')
        ax[0].grid(True)
        
    if 'kl_loss_epoch' in df.columns:
        df['kl_loss_epoch'].dropna().reset_index(drop=True).plot(ax=ax[1], marker='o')
        ax[1].set_title('KL Loss')
        ax[1].grid(True)
        
    if 'val_loss' in df.columns:
        df['val_loss'].dropna().reset_index(drop=True).plot(ax=ax[2], marker='o')
        ax[2].set_title('Validation Loss')
        ax[2].grid(True)
        
    if 'val_psnr' in df.columns:
        df['val_psnr'].dropna().reset_index(drop=True).plot(ax=ax[3], marker='o')
        ax[3].set_title('Validation PSNR')
        ax[3].grid(True)
        
    plt.tight_layout()
    plt.show()

print("Now we can move onto the next step, we will do predictions in the 03JUMP_predict notebook.")

