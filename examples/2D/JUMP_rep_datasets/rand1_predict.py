#!/usr/bin/env python
"""
MicroSplit Prediction Script for HPC
Generates predictions from trained MicroSplit model on test data
"""

import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for HPC
import matplotlib.pyplot as plt
import tifffile
import torch
from pathlib import Path
import pandas as pd
import argparse

# CAREamics imports
from careamics.lightning import VAEModule
from careamics.lvae_training.dataset import DataSplitType

# MicroSplit reproducibility imports
from microsplit_reproducibility.configs.factory import (
    create_algorithm_config,
    get_likelihood_config,
    get_loss_config,
    get_model_config,
    get_optimizer_config,
    get_training_config,
    get_lr_scheduler_config,
)
from microsplit_reproducibility.utils.io import load_checkpoint, load_checkpoint_path
from microsplit_reproducibility.datasets import create_train_val_datasets
from microsplit_reproducibility.utils.paper_metrics import avg_range_inv_psnr, structural_similarity
from microsplit_reproducibility.configs.data.JUMP import get_data_configs
from microsplit_reproducibility.configs.parameters.JUMP import get_microsplit_parameters
from microsplit_reproducibility.datasets.JUMP import get_train_val_data
from microsplit_reproducibility.notebook_utils.JUMP import (
    load_pretrained_model,
    get_unnormalized_predictions,
    get_target,
    get_input,
    full_frame_evaluation,
    show_sampling,
    pick_random_patches_with_content
)


class Channels:
    DNA = "DNA"
    RNA = "RNA"
    ER = "ER"
    AGP = "AGP"
    Mito = "Mito"


def setup_paths(test_data_path=None):
    """Configure all paths for the experiment"""
    if test_data_path is None:
        # Default to bio-rand-1 test split
        dataset_dir = "/home/diya.srivastava/Desktop/repos/JUMP-MicroSplit/examples/2D/JUMP_rep_datasets/rand-1/5_channels/dna_rna_er_agp_mito"
    else:
        dataset_dir = test_data_path
    
    paths = {
        'dataset_dir': dataset_dir,
        'noise_model_path': "/home/diya.srivastava/Desktop/repos/JUMP-MicroSplit/examples/2D/JUMP_rep_datasets/rand-1-noise_models",
        'checkpoint_path': "/home/diya.srivastava/Desktop/repos/JUMP-MicroSplit/examples/2D/JUMP_rep_datasets/checkpoints/rand-1-v2/best-epoch=35.ckpt"
    }
    return paths

def initialize_model(paths, target_channel_list):
    """Initialize model with trained checkpoint"""
    print("Initializing model...")
    
    # Get data configurations
    train_data_config, val_data_config, test_data_config = get_data_configs(
        channel_idx_list=target_channel_list,
    )
    
    # IMPORTANT: Keep train as Train type (required by dataset class)
    # But set test to use ALL data from test_data directory
    train_data_config.datasplit_type = DataSplitType.Train  # Keep as Train!
    val_data_config.datasplit_type = DataSplitType.Val      # Keep as Val!
    test_data_config.datasplit_type = DataSplitType.All     # Use all test data
    
    # Load datasets - but we only care about test_dset
    train_dset, val_dset, test_dset, data_stats = create_train_val_datasets(
        datapath=paths['dataset_dir'],
        train_config=train_data_config,
        val_config=val_data_config,
        test_config=test_data_config,
        load_data_func=get_train_val_data,
    )
    
    # Fix dataset paths (workaround for missing _fpath attribute)
    for dset in [test_dset, val_dset]:
        if hasattr(dset, 'data_dir') and not hasattr(dset, '_fpath'):
            dset._fpath = Path(dset.data_dir)
        elif not hasattr(dset, '_fpath'):
            dset._fpath = Path(paths['dataset_dir'])
    
    # Set up experiment parameters
    experiment_params = get_microsplit_parameters(
        nm_path=paths['noise_model_path'],
        channel_idx_list=target_channel_list,
    )
    experiment_params["data_stats"] = data_stats
    
    # Configure model components
    model_config = get_model_config(**experiment_params)
    loss_config = get_loss_config(**experiment_params)
    gaussian_lik_config, noise_model_config, nm_lik_config = get_likelihood_config(**experiment_params)
    
    # Create algorithm config
    experiment_config = create_algorithm_config(
        algorithm=experiment_params["algorithm"],
        loss_config=loss_config,
        model_config=model_config,
        gaussian_lik_config=gaussian_lik_config,
        nm_config=noise_model_config,
        nm_lik_config=nm_lik_config,
    )
    
    # Create and load model
    model = VAEModule(algorithm_config=experiment_config)
    load_pretrained_model(model, paths['checkpoint_path'])
    
    print(f"Model loaded from {paths['checkpoint_path']}")
    print(f"Test dataset contains {test_dset.get_num_frames()} frames")
    
    return model, test_dset, data_stats

def generate_predictions(model, dset, target_channel_list, mmse_count=50, num_workers=2, batch_size=8):
    """Generate predictions on test data"""
    print("Generating predictions...")
    
    # Ensure _fpath is Path object
    dset._fpath = Path(dset._fpath)
    
    stitched_predictions, norm_stitched_predictions, stitched_stds = get_unnormalized_predictions(
        model=model,
        dset=dset,
        target_channel_list=target_channel_list,
        mmse_count=mmse_count,
        num_workers=num_workers,
        batch_size=batch_size
    )
    
    tar = get_target(dset)
    inp = get_input(dset)
    
    print(f"Predictions generated for {len(stitched_predictions)} frames")
    
    return stitched_predictions, tar, inp, stitched_stds


def visualize_full_frame(stitched_predictions, tar, inp, save_dir, frame_idx=0):
    """Create full frame evaluation visualization"""
    print(f"Creating full frame visualization for frame {frame_idx}...")
    
    fig = full_frame_evaluation(stitched_predictions[frame_idx], tar[frame_idx], inp[frame_idx])
    output_path = os.path.join(save_dir, f"full_frame_evaluation_frame{frame_idx}.png")
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    print(f"Saved: {output_path}")


def visualize_random_patches(stitched_predictions, tar, inp, target_channel_list, save_dir, img_sz=128, nrows=5):
    """Create detailed view of random regions with content"""
    print("Creating random patch visualizations...")
    
    rand_locations = pick_random_patches_with_content(tar, img_sz)
    ncols = 2 * len(target_channel_list) + 1
    nrows = min(len(rand_locations), nrows)
    
    fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=(ncols * 3, nrows * 3))
    
    for i, (h_start, w_start) in enumerate(rand_locations[:nrows]):
        ax[i, 0].imshow(inp[0, h_start:h_start+img_sz, w_start:w_start+img_sz])
        
        for j in range(ncols // 2):
            vmin = stitched_predictions[0, h_start:h_start+img_sz, w_start:w_start+img_sz, j].min()
            vmax = stitched_predictions[0, h_start:h_start+img_sz, w_start:w_start+img_sz, j].max()
            
            ax[i, 2*j+1].imshow(tar[0, h_start:h_start+img_sz, w_start:w_start+img_sz, j], 
                               vmin=vmin, vmax=vmax)
            ax[i, 2*j+2].imshow(stitched_predictions[0, h_start:h_start+img_sz, w_start:w_start+img_sz, j], 
                               vmin=vmin, vmax=vmax)
    
    ax[0, 0].set_title('Primary Input')
    for i in range(len(target_channel_list)):
        ax[0, 2*i+1].set_title(f'Target Channel {i+1}')
        ax[0, 2*i+2].set_title(f'Predicted Channel {i+1}')
    
    plt.subplots_adjust(wspace=0.03, hspace=0.03)
    for a in ax.ravel():
        a.set_xticks([])
        a.set_yticks([])
    
    plt.tight_layout()
    output_path = os.path.join(save_dir, "random_patches_comparison.png")
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    print(f"Saved: {output_path}")


def visualize_specific_region(stitched_predictions, tar, inp, target_channel_list, save_dir, 
                              y_start=600, x_start=600, crop_size=128):
    """Create visualization of specific region"""
    print(f"Creating specific region visualization at ({y_start}, {x_start})...")
    
    ncols = len(target_channel_list) + 1
    nrows = 2
    fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=(ncols * 5, nrows * 5))
    
    # Input image
    ax[0, 0].imshow(inp[0, y_start:y_start+crop_size, x_start:x_start+crop_size])
    ax[1, 0].axis('off')
    ax[0, 0].set_title("Input")
    
    # Display each channel
    for i in range(len(target_channel_list)):
        vmin = stitched_predictions[0, y_start:y_start+crop_size, x_start:x_start+crop_size, i].min()
        vmax = stitched_predictions[0, y_start:y_start+crop_size, x_start:x_start+crop_size, i].max()
        
        # Target
        ax[0, i+1].imshow(tar[0, y_start:y_start+crop_size, x_start:x_start+crop_size, i], 
                         vmin=vmin, vmax=vmax)
        ax[0, i+1].set_title(f"Channel {i+1} ({target_channel_list[i]})")
        
        # Prediction
        ax[1, i+1].imshow(stitched_predictions[0, y_start:y_start+crop_size, x_start:x_start+crop_size, i], 
                         vmin=vmin, vmax=vmax)
        
        # Add labels to last channel
        if i == len(target_channel_list) - 1:
            ax[0, i+1].yaxis.set_label_position("right")
            ax[0, i+1].set_ylabel("Target")
            ax[1, i+1].yaxis.set_label_position("right")
            ax[1, i+1].set_ylabel("Predicted")
    
    plt.tight_layout()
    output_path = os.path.join(save_dir, f"specific_region_y{y_start}_x{x_start}.png")
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    print(f"Saved: {output_path}")


def visualize_posterior_sampling(dset, model, target_channel_list, save_dir, num_examples=3):
    """Create posterior sampling visualization"""
    print("Creating posterior sampling visualization...")
    
    imgsz = 3
    ncols = 6
    num_channels = len(target_channel_list)
    
    fig, ax = plt.subplots(figsize=(imgsz*ncols, imgsz*num_channels*num_examples),
                          ncols=ncols, nrows=num_channels*num_examples)
    
    for i in range(num_examples):
        row_indices = slice(i*num_channels, (i+1)*num_channels)
        show_sampling(dset, model, ax=ax[row_indices])
    
    plt.tight_layout()
    output_path = os.path.join(save_dir, "posterior_sampling.png")
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    print(f"Saved: {output_path}")


def calculate_metrics(tar, stitched_predictions, target_channel_list):
    """Calculate PSNR and SSIM metrics"""
    print("Calculating quantitative metrics...")
    
    psnr_results = []
    ssim_results = []
    
    for ch_idx in range(tar.shape[-1]):
        # Calculate PSNR
        psnr_val = avg_range_inv_psnr(
            [tar[i, ..., ch_idx] for i in range(tar.shape[0])],
            [stitched_predictions[i, ..., ch_idx] for i in range(stitched_predictions.shape[0])]
        )
        psnr_results.append(psnr_val)
        
        # Calculate SSIM
        ssim_vals = []
        for i in range(tar.shape[0]):
            ssim_vals.append(structural_similarity(
                tar[i, ..., ch_idx],
                stitched_predictions[i, ..., ch_idx],
                data_range=tar[i, ..., ch_idx].max() - tar[i, ..., ch_idx].min()
            ))
        ssim_results.append((np.mean(ssim_vals), np.std(ssim_vals) / np.sqrt(len(ssim_vals))))
    
    return psnr_results, ssim_results


def save_metrics(psnr_results, ssim_results, target_channel_list, save_dir):
    """Save quantitative metrics to CSV"""
    print("Saving metrics...")
    
    # Print results
    print("\nQuantitative Evaluation:")
    print("=======================")
    for i, channel in enumerate(target_channel_list):
        print(f"Channel {i+1} ({channel}):")
        print(f"  PSNR: {psnr_results[i][0]:.2f} ± {psnr_results[i][1]:.3f}")
        print(f"  SSIM: {ssim_results[i][0]:.4f} ± {ssim_results[i][1]:.4f}")
    
    # Save to CSV
    metrics_data = []
    for i, channel in enumerate(target_channel_list):
        metrics_data.append({
            'Channel': channel,
            'PSNR_Mean': psnr_results[i][0],
            'PSNR_Std': psnr_results[i][1],
            'SSIM_Mean': ssim_results[i][0],
            'SSIM_Std': ssim_results[i][1]
        })
    
    metrics_df = pd.DataFrame(metrics_data)
    metrics_csv_path = os.path.join(save_dir, "quantitative_metrics.csv")
    metrics_df.to_csv(metrics_csv_path, index=False)
    
    print(f"\nQuantitative metrics saved to: {metrics_csv_path}")


def save_predictions(stitched_predictions, tar, inp, target_channel_list, save_dir):
    """Save all predictions, targets, and inputs as TIFF files"""
    print("Saving prediction TIFFs...")
    
    for i in range(len(stitched_predictions)):
        for j, channel in enumerate(target_channel_list):
            # Save prediction
            pred_filename = f"test_pred_frame{i}_{channel}.tif"
            tifffile.imwrite(
                os.path.join(save_dir, pred_filename),
                stitched_predictions[i, ..., j].astype(np.float32)
            )
            
            # Save target
            target_filename = f"target_frame{i}_{channel}.tif"
            tifffile.imwrite(
                os.path.join(save_dir, target_filename),
                tar[i, ..., j].astype(np.float32)
            )
        
        # Save input
        input_filename = f"input_frame{i}_combined.tif"
        tifffile.imwrite(
            os.path.join(save_dir, input_filename),
            inp[i].astype(np.float32)
        )
    
    print(f"All predictions saved to: {save_dir}")


def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(description='MicroSplit Prediction')
    parser.add_argument('--test-data', type=str, 
                       default="/home/diya.srivastava/Desktop/repos/JUMP-MicroSplit/examples/2D/JUMP_rep_datasets/test_data/5_channels/dna_rna_er_agp_mito",
                       help='Path to test dataset')
    args = parser.parse_args()
    
    print("=" * 80)
    print("MicroSplit Prediction Pipeline - HPC Version")
    print("=" * 80)
    
    # Configuration
    TARGET_CHANNEL_LIST = [Channels.DNA, Channels.RNA, Channels.ER, Channels.AGP, Channels.Mito]
    print(f"\nSelected channels: {TARGET_CHANNEL_LIST}")
    
    # Setup paths with custom test data
    paths = setup_paths(test_data_path=args.test_data)
    
    # Create output directory
    channel_names = "_".join(ch for ch in TARGET_CHANNEL_LIST)
    save_dir = f"rand1_predictions_v2"
    os.makedirs(save_dir, exist_ok=True)
    print(f"\nOutput directory: {save_dir}")
    
    # Initialize model
    model, test_dset, data_stats = initialize_model(paths, TARGET_CHANNEL_LIST)
    print(f"Using test data (containing {test_dset.get_num_frames()} frames)")
    
    # Generate predictions
    stitched_predictions, tar, inp, stitched_stds = generate_predictions(
        model=model,
        dset=test_dset,
        target_channel_list=TARGET_CHANNEL_LIST,
        mmse_count=50,
        num_workers=2,
        batch_size=8
    )
    
    # Create visualizations
    print("\n" + "=" * 80)
    print("Generating Visualizations")
    print("=" * 80)
    
    visualize_full_frame(stitched_predictions, tar, inp, save_dir, frame_idx=0)
    visualize_random_patches(stitched_predictions, tar, inp, TARGET_CHANNEL_LIST, save_dir)
    visualize_specific_region(stitched_predictions, tar, inp, TARGET_CHANNEL_LIST, save_dir)
    visualize_posterior_sampling(test_dset, model, TARGET_CHANNEL_LIST, save_dir)
    
    # Calculate and save metrics
    print("\n" + "=" * 80)
    print("Quantitative Evaluation")
    print("=" * 80)
    
    psnr_results, ssim_results = calculate_metrics(tar, stitched_predictions, TARGET_CHANNEL_LIST)
    save_metrics(psnr_results, ssim_results, TARGET_CHANNEL_LIST, save_dir)
    
    # Save predictions
    print("\n" + "=" * 80)
    print("Saving Predictions")
    print("=" * 80)
    
    save_predictions(stitched_predictions, tar, inp, TARGET_CHANNEL_LIST, save_dir)
    
    print("\n" + "=" * 80)
    print("Prediction process complete!")
    print("=" * 80)
    print(f"\nAll results saved to: {save_dir}")


if __name__ == "__main__":
    main()