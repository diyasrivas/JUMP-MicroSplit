#!/usr/bin/env python3
import os
import sys
import argparse
import glob
import numpy as np
import matplotlib.pyplot as plt
import tifffile
import torch
import pandas as pd
from pathlib import Path
from tqdm import tqdm

# Import CAREamics libraries
from careamics.lightning import VAEModule
from careamics.lvae_training.dataset import DataSplitType

# Import microsplit_reproducibility modules
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

# Import JUMP-specific modules
from microsplit_reproducibility.configs.data.JUMP import get_data_configs
from microsplit_reproducibility.configs.parameters.JUMP import get_microsplit_parameters
from microsplit_reproducibility.datasets.JUMP import get_train_val_data
from microsplit_reproducibility.notebook_utils.JUMP import (
    load_pretrained_model,
    get_target,
    get_input,
    full_frame_evaluation,
    show_sampling,
    pick_random_patches_with_content
)

def get_unnormalized_predictions(model, dset, target_channel_list, mmse_count=10, num_workers=4, batch_size=8):
    """Custom implementation that doesn't rely on problematic CAREamics internals"""
    # Create dataloader
    dataloader = torch.utils.data.DataLoader(
        dset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False
    )
    
    # Predict in batches
    all_predictions = []
    all_stds = []
    
    print("Predicting tiles:")
    with torch.no_grad():
        for batch in tqdm(dataloader):
            if isinstance(batch, tuple) and len(batch) == 2:
                inputs, _ = batch
            else:
                inputs = batch
                
            # Make sure inputs is a tensor
            if not isinstance(inputs, torch.Tensor):
                if isinstance(inputs, list):
                    inputs = torch.tensor(inputs)
                else:
                    # This handles other types like numpy arrays
                    inputs = torch.from_numpy(np.array(inputs))
                    
            inputs = inputs.to(model.device)
                
            # Generate multiple samples for MMSE
            batch_predictions = []
            for _ in range(mmse_count):
                pred_batch, _ = model(inputs)
                batch_predictions.append(pred_batch.cpu().numpy())
                
            # Calculate MMSE and std
            batch_predictions = np.stack(batch_predictions, axis=1)
            mmse_prediction = np.mean(batch_predictions, axis=1)
            std_prediction = np.std(batch_predictions, axis=1)
            
            all_predictions.append(mmse_prediction)
            all_stds.append(std_prediction)

def save_figure(fig, filename):
    """Save matplotlib figure with high resolution."""
    fig.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close(fig)

def predict_and_evaluate(dataset_dir, checkpoint_dir, output_dir, channels, noise_models_dir, reduced_data=True, mmse_count=100):
    """
    Run prediction and evaluation for a specific channel combination.
    
    Parameters:
    -----------
    dataset_dir : str
        Path to the dataset directory
    checkpoint_dir : str
        Path to the checkpoint directory
    output_dir : str
        Path to save results
    channels : list
        List of channel names
    noise_models_dir : str
        Path to noise models directory
    reduced_data : bool
        Whether to use reduced data for quick testing
    mmse_count : int
        Number of samples for MMSE estimation
    """
    print(f"\n{'='*80}")
    print(f"Processing dataset: {os.path.basename(dataset_dir)}")
    print(f"Channels: {channels}")
    print(f"{'='*80}")
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Get data configurations
    train_data_config, val_data_config, test_data_config = get_data_configs(
        channel_idx_list=channels,
    )
    
    # Set up MicroSplit parameters
    experiment_params = get_microsplit_parameters(
        nm_path=noise_models_dir,  # Path to noise models
        channel_idx_list=channels,
    )
    
    # Load datasets
    train_dset, val_dset, test_dset, data_stats = create_train_val_datasets(
        datapath=dataset_dir,
        train_config=train_data_config,
        val_config=val_data_config,
        test_config=test_data_config,
        load_data_func=get_train_val_data,
    )
    
    # Choose whether to evaluate on validation or test data
    evaluate_on_validation_data = False
    if evaluate_on_validation_data:
        print('Using validation data', end='')
        dset = val_dset
    else:
        print('Using test data', end='')
        dset = test_dset
    print(f' (containing a total of {dset.get_num_frames()} frames).')
    
    # Optional: Reduce dataset size for quick testing
    if reduced_data:
        print("Using REDUCED evaluation data for quick testing!")
        dset.reduce_data([0])
    else:
        print('Using the full set of evaluation data!')
        print(f'(Using {dset.get_num_frames()} frames for evaluations.)')
    
    # Find the best checkpoint
    checkpoint_path = find_best_checkpoint(checkpoint_dir)
    if not checkpoint_path:
        raise ValueError(f"No checkpoint found in {checkpoint_dir}")
    print(f"Using checkpoint: {checkpoint_path}")
    
    # Initialize the model
    experiment_params["data_stats"] = data_stats
    model_config = get_model_config(**experiment_params)
    loss_config = get_loss_config(**experiment_params)
    gaussian_lik_config, noise_model_config, nm_lik_config = get_likelihood_config(
        **experiment_params
    )
    
    # Create algorithm config
    experiment_config = create_algorithm_config(
        algorithm=experiment_params["algorithm"],
        loss_config=loss_config,
        model_config=model_config,
        gaussian_lik_config=gaussian_lik_config,
        nm_config=noise_model_config,
        nm_lik_config=nm_lik_config,
    )
    
    # Create model and load checkpoint
    model = VAEModule(algorithm_config=experiment_config)
    load_pretrained_model(model, checkpoint_path)
    
    # Generate predictions
    stitched_predictions, norm_stitched_predictions, stitched_stds = get_unnormalized_predictions(
        model=model,
        dset=dset,
        target_channel_list=channels,
        mmse_count=mmse_count,
        num_workers=4,
        batch_size=8
    )
    
    # Get target and input data for evaluation
    tar = get_target(dset)
    inp = get_input(dset)
    
    # PART 1: VISUALIZATIONS
    
    # 1. Overview: full frame evaluation
    frame_idx = 0
    assert frame_idx < len(stitched_predictions), f"Frame index {frame_idx} out of bounds"
    fig = full_frame_evaluation(stitched_predictions[frame_idx], tar[frame_idx], inp[frame_idx])
    save_figure(plt.gcf(), os.path.join(output_dir, "full_frame_evaluation.png"))
    
    # 2. Detailed view of random regions with content
    img_sz = 128
    rand_locations = pick_random_patches_with_content(tar, img_sz)
    ncols = 2*len(channels) + 1
    nrows = min(len(rand_locations), 5)
    fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=(ncols * 3, nrows * 3))
    
    for i, (h_start, w_start) in enumerate(rand_locations[:nrows]):
        ax[i,0].imshow(inp[0,h_start:h_start+img_sz, w_start:w_start+img_sz])
        for j in range(ncols//2):
            vmin = stitched_predictions[0,h_start:h_start+img_sz, w_start:w_start+img_sz,j].min()
            vmax = stitched_predictions[0,h_start:h_start+img_sz, w_start:w_start+img_sz,j].max()
            ax[i,2*j+1].imshow(tar[0,h_start:h_start+img_sz, w_start:w_start+img_sz,j], vmin=vmin, vmax=vmax)
            ax[i,2*j+2].imshow(stitched_predictions[0,h_start:h_start+img_sz, w_start:w_start+img_sz,j], vmin=vmin, vmax=vmax)
    
    ax[0,0].set_title('Primary Input')
    for i in range(len(channels)):
        ax[0,2*i+1].set_title(f'Target Channel {i+1} ({channels[i]})')
        ax[0,2*i+2].set_title(f'Predicted Channel {i+1} ({channels[i]})')
    
    # Reduce spacing between subplots
    plt.subplots_adjust(wspace=0.03, hspace=0.03)
    
    # Remove axis ticks for all subplots
    for a in ax.ravel():
        a.set_xticks([])
        a.set_yticks([])
    
    plt.tight_layout()
    save_figure(fig, os.path.join(output_dir, "random_patches.png"))
    
    # 3. Inspect a specific region
    y_start = 600
    x_start = 600
    crop_size = 128
    ncols = len(channels) + 1
    nrows = 2
    fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=(ncols * 5, nrows * 5))
    
    # Input image
    ax[0,0].imshow(inp[0,y_start:y_start+crop_size, x_start:x_start+crop_size])
    ax[1,0].axis('off')
    ax[0,0].set_title("Input")
    
    # Display each channel
    for i in range(len(channels)):
        vmin = stitched_predictions[0,y_start:y_start+crop_size, x_start:x_start+crop_size,i].min()
        vmax = stitched_predictions[0,y_start:y_start+crop_size, x_start:x_start+crop_size,i].max()
        
        # Target
        ax[0,i+1].imshow(tar[0,y_start:y_start+crop_size, x_start:x_start+crop_size,i], vmin=vmin, vmax=vmax)
        ax[0,i+1].set_title(f"Channel {i+1} ({channels[i]})")
        
        # Prediction
        ax[1,i+1].imshow(stitched_predictions[0,y_start:y_start+crop_size, x_start:x_start+crop_size,i], vmin=vmin, vmax=vmax)
    
    # Add labels to the last channel
    ax[0,-1].yaxis.set_label_position("right")
    ax[0,-1].set_ylabel("Target")
    ax[1,-1].yaxis.set_label_position("right")
    ax[1,-1].set_ylabel("Predicted")
    
    plt.tight_layout()
    save_figure(fig, os.path.join(output_dir, "specific_region.png"))
    
    # 4. Posterior sampling
    imgsz = 3
    ncols = 6
    num_channels = len(channels)
    num_examples = 2  # Reduced for speed
    
    # Create figure with enough rows for all channels across multiple examples
    fig, ax = plt.subplots(
        figsize=(imgsz*ncols, imgsz*num_channels*num_examples),
        ncols=ncols, 
        nrows=num_channels*num_examples
    )
    
    # Show examples, with each example having num_channels rows
    for i in range(num_examples):
        row_indices = slice(i*num_channels, (i+1)*num_channels)
        show_sampling(dset, model, ax=ax[row_indices])
    
    plt.tight_layout()
    save_figure(fig, os.path.join(output_dir, "posterior_sampling.png"))
    
    # PART 2: QUANTITATIVE EVALUATION
    
    # Calculate metrics
    psnr_results = []
    ssim_results = []
    
    for ch_idx in range(tar.shape[-1]):
        # Calculate PSNR for this channel
        psnr_val = avg_range_inv_psnr(
            [tar[i,...,ch_idx] for i in range(tar.shape[0])],
            [stitched_predictions[i,...,ch_idx] for i in range(stitched_predictions.shape[0])]
        )
        psnr_results.append(psnr_val)
        
        # Calculate SSIM for this channel
        ssim_vals = []
        for i in range(tar.shape[0]):
            ssim_vals.append(structural_similarity(
                tar[i,...,ch_idx],
                stitched_predictions[i,...,ch_idx],
                data_range=tar[i,...,ch_idx].max() - tar[i,...,ch_idx].min()
            ))
        ssim_results.append((np.mean(ssim_vals), np.std(ssim_vals)/np.sqrt(len(ssim_vals))))
    
    # Print results
    print("\nQuantitative Evaluation:")
    print("=======================")
    metrics_data = []
    
    for i, channel in enumerate(channels):
        print(f"Channel {i+1} ({channel}):")
        print(f" PSNR: {psnr_results[i][0]:.2f} ± {psnr_results[i][1]:.3f}")
        print(f" SSIM: {ssim_results[i][0]:.4f} ± {ssim_results[i][1]:.4f}")
        
        # Add metrics to data list
        metrics_data.append({
            'Channel': channel,
            'PSNR_Mean': psnr_results[i][0],
            'PSNR_Std': psnr_results[i][1],
            'SSIM_Mean': ssim_results[i][0],
            'SSIM_Std': ssim_results[i][1]
        })
    
    # Save metrics to CSV
    metrics_df = pd.DataFrame(metrics_data)
    metrics_csv_path = os.path.join(output_dir, "quantitative_metrics.csv")
    metrics_df.to_csv(metrics_csv_path, index=False)
    print(f"\nQuantitative metrics saved to: {metrics_csv_path}")
    
    # PART 3: SAVE PREDICTIONS AS TIFF FILES
    
    for i in range(len(stitched_predictions)):
        for j, channel in enumerate(channels):
            # Save prediction
            pred_filename = f"pred_frame{i}_{channel}.tif"
            tifffile.imwrite(
                os.path.join(output_dir, pred_filename),
                stitched_predictions[i, ..., j].astype(np.float32)
            )
            
            # Save target for reference
            target_filename = f"target_frame{i}_{channel}.tif"
            tifffile.imwrite(
                os.path.join(output_dir, target_filename),
                tar[i, ..., j].astype(np.float32)
            )
        
        # Save input image for reference
        input_filename = f"input_frame{i}_combined.tif"
        tifffile.imwrite(
            os.path.join(output_dir, input_filename),
            inp[i].astype(np.float32)
        )
    
    print(f"\nAll predictions and metrics saved to directory: {output_dir}")
    print("\nPrediction process complete!")

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Run μSplit predictions on JUMP datasets')
    parser.add_argument('--repo-dir', type=str, required=True,
                        help='Path to the repository directory (e.g., /home/diya.srivastava/Desktop/repos/JUMP-MicroSplit)')
    parser.add_argument('--num-channels', type=int, default=2,
                        help='Number of channels (2 or 5)')
    parser.add_argument('--dataset', type=str, default=None,
                        help='Specific dataset to process (e.g., "dna_mito"). If not provided, all datasets are processed.')
    parser.add_argument('--reduced', action='store_true', default=True,
                        help='Use reduced dataset for faster processing')
    parser.add_argument('--mmse-count', type=int, default=100,
                        help='Number of samples for MMSE estimation')
    
    args = parser.parse_args()
    
    # Construct absolute paths
    base_dir = os.path.join(args.repo_dir, "examples/2D/JUMP")
    experiments_dir = os.path.join(base_dir, "experiments") 
    checkpoints_dir = os.path.join(base_dir, "checkpoints")
    noise_models_dir = os.path.join(base_dir, "noise_models")
    results_dir = os.path.join(base_dir, "results")
    
    # Create results directory if it doesn't exist
    os.makedirs(results_dir, exist_ok=True)
    
    # Define all available channels
    class Channels:
        DNA = "DNA"
        Mito = "Mito"
        RNA = "RNA"
        ER = "ER"
        AGP = "AGP"
    
    ALL_CHANNELS = [Channels.DNA, Channels.RNA, Channels.ER, Channels.AGP, Channels.Mito]
    
    # Define datasets to process
    if args.num_channels == 5:
        # 5-channel datasets
        datasets = [
            {
                'name': 'dna_rna_er_agp_mito',
                'channels': [Channels.DNA, Channels.RNA, Channels.ER, Channels.AGP, Channels.Mito]
            }
        ]
    elif args.num_channels == 2:
        # 2-channel datasets
        datasets = [
            {'name': 'dna_rna', 'channels': [Channels.DNA, Channels.RNA]},
            {'name': 'dna_er', 'channels': [Channels.DNA, Channels.ER]},
            {'name': 'dna_agp', 'channels': [Channels.DNA, Channels.AGP]},
            {'name': 'dna_mito', 'channels': [Channels.DNA, Channels.Mito]},
            {'name': 'rna_er', 'channels': [Channels.RNA, Channels.ER]},
            {'name': 'rna_agp', 'channels': [Channels.RNA, Channels.AGP]},
            {'name': 'rna_mito', 'channels': [Channels.RNA, Channels.Mito]},
            {'name': 'er_agp', 'channels': [Channels.ER, Channels.AGP]},
            {'name': 'er_mito', 'channels': [Channels.ER, Channels.Mito]},
            {'name': 'agp_mito', 'channels': [Channels.AGP, Channels.Mito]}
        ]
    else:
        raise ValueError(f"Unsupported number of channels: {args.num_channels}")
    
    # If a specific dataset is provided, filter the list
    if args.dataset:
        datasets = [d for d in datasets if d['name'] == args.dataset]
        if not datasets:
            raise ValueError(f"Dataset '{args.dataset}' not found")
    
    # Process each dataset
    for dataset in datasets:
        dataset_name = dataset['name']
        channels = dataset['channels']
        
        # Construct paths with absolute references
        dataset_dir = os.path.join(experiments_dir, f"{args.num_channels}_channels", dataset_name)
        checkpoint_dir = os.path.join(checkpoints_dir, f"{args.num_channels}_channels", dataset_name)
        output_dir = os.path.join(results_dir, f"{args.num_channels}channels_predictions_{dataset_name}")
        
        # Verify paths exist
        if not os.path.exists(dataset_dir):
            print(f"WARNING: Dataset directory not found: {dataset_dir}")
            continue
            
        if not os.path.exists(checkpoint_dir):
            print(f"WARNING: Checkpoint directory not found: {checkpoint_dir}")
            continue
        
        # Run prediction and evaluation
        try:
            predict_and_evaluate(
                dataset_dir=dataset_dir,
                checkpoint_dir=checkpoint_dir,
                output_dir=output_dir,
                channels=channels,
                noise_models_dir=noise_models_dir,
                reduced_data=args.reduced,
                mmse_count=args.mmse_count
            )
        except Exception as e:
            print(f"ERROR processing dataset {dataset_name}: {str(e)}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    main()
