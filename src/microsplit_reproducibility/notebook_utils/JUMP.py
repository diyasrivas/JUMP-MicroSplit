import matplotlib.pyplot as plt
import numpy as np
import torch
import os
from careamics.lvae_training.eval_utils import get_predictions
from careamics.lightning import VAEModule
from microsplit_reproducibility.datasets import SplittingDataset

def load_pretrained_model(model: VAEModule, ckpt_path):
    ckpt_dict = torch.load(ckpt_path)
    ckpt_dict['state_dict'] = {'model.' + k: v for k, v in ckpt_dict['state_dict'].items()}
    model.load_state_dict(ckpt_dict['state_dict'], strict=False)
    print(f"Loaded model from {ckpt_path}")

def get_all_channel_list(target_channel_list):
    """
    Just return target channel list with combined channel index
    For JUMP, we just append the combined channel at the end
    """
    return target_channel_list + ["combined"]

def get_unnormalized_predictions(model: VAEModule, dset: SplittingDataset, dataset_name, target_channel_list, 
                               mmse_count, num_workers=4, grid_size=32, batch_size=8):
    """
    Get the stitched predictions which have been unnormalized.
    
    Parameters:
    -----------
    model: VAEModule
        The trained model
    dset: SplittingDataset
        Dataset containing the images
    dataset_name: str
        Name of the dataset, serves as key for dictionary results 
        (analogous to exposure_duration in HT_LIF24)
    target_channel_list: list
        List of target channels to predict
    mmse_count: int
        Number of samples for MMSE estimation
    num_workers: int
        Number of worker processes for data loading
    grid_size: int
        Size of grid for tiled predictions
    batch_size: int
        Batch size for predictions
        
    Returns:
    --------
    tuple of (unnormalized predictions, normalized predictions, standard deviations)
    """
    # You might need to adjust the batch size depending on the available memory
    stitched_predictions, stitched_stds = get_predictions(
        model=model,
        dset=dset,
        batch_size=batch_size,
        num_workers=num_workers,
        mmse_count=mmse_count,
        tile_size=model.model.image_size,
        grid_size=grid_size,
    )
    
    # Check if results are in dictionary format (matching HT_LIF24 behavior)
    if isinstance(stitched_predictions, dict):
        if dataset_name in stitched_predictions:
            stitched_predictions = stitched_predictions[dataset_name]
            stitched_stds = stitched_stds[dataset_name]
        else:
            # Fall back to first value if dataset_name not found
            stitched_predictions = list(stitched_predictions.values())[0]
            stitched_stds = list(stitched_stds.values())[0]
    
    # Handle channel selection if needed
    stitched_predictions = stitched_predictions[...,:len(target_channel_list)]
    stitched_stds = stitched_stds[...,:len(target_channel_list)]
    
    # Denormalize the predictions
    mean_params, std_params = dset.get_mean_std()
    unnorm_stitched_predictions = stitched_predictions*std_params['target'].squeeze().reshape(1,1,1,-1) + mean_params['target'].squeeze().reshape(1,1,1,-1)
    
    return unnorm_stitched_predictions, stitched_predictions, stitched_stds

def get_unnormalized_predictions(model: VAEModule, dset: SplittingDataset, target_channel_list, mmse_count, 
                                num_workers=4, grid_size=32, batch_size=8):
    """
    Get the stitched predictions which have been unnormalized.
    """
    # You might need to adjust the batch size depending on the available memory
    stitched_predictions, stitched_stds = get_predictions(
        model=model,
        dset=dset,
        batch_size=batch_size,
        num_workers=num_workers,
        mmse_count=mmse_count,
        tile_size=model.model.image_size,
        grid_size=grid_size,
    )
    
    # For JUMP, we don't need the exposure_duration dictionary level
    # Just get first key if dictionary is returned
    if isinstance(stitched_predictions, dict):
        stitched_predictions = list(stitched_predictions.values())[0]
        stitched_stds = list(stitched_stds.values())[0]

    # Handle channel selection if needed
    stitched_predictions = stitched_predictions[...,:len(target_channel_list)]
    stitched_stds = stitched_stds[...,:len(target_channel_list)]
    
    # Denormalize the predictions
    mean_params, std_params = dset.get_mean_std()
    unnorm_stitched_predictions = stitched_predictions*std_params['target'].squeeze().reshape(1,1,1,-1) + mean_params['target'].squeeze().reshape(1,1,1,-1)
    
    return unnorm_stitched_predictions, stitched_predictions, stitched_stds

def get_target(dset):
    """Get target channels from dataset"""
    return dset._data[...,:-1].copy()

def get_input(dset):
    """Get input (combined) channel from dataset"""
    return dset._data[...,-1].copy()

def pick_random_patches_with_content(tar, patch_size):    
    """Find random patches with interesting content based on standard deviation"""
    H, W = tar.shape[1:3]
    std_patches = []
    indices = []
    for i in range(1000):
        h_start = np.random.randint(H - patch_size)
        w_start = np.random.randint(W - patch_size)
        std_tmp= []
        for ch_idx in range(tar.shape[-1]):
            std_tmp.append(tar[0,h_start:h_start+patch_size,w_start:w_start+patch_size,ch_idx].std())
        
        std_patches.append(np.mean(std_tmp))
        indices.append((h_start,w_start))
    
    # sort by std
    indices = np.array(indices)[np.argsort(std_patches)][-40:]

    distances = np.linalg.norm(indices[:,None] - indices[None], axis=-1)
    # pick the indices of the indices that are at least patch_size pixels apart
    final_indices = [0]
    for i in range(1, len(indices)):
        if np.all(distances[i,final_indices] >= patch_size):
            final_indices.append(i)

    final_indices = indices[final_indices,:]
    return final_indices

def pick_random_inputs_with_content(dset):
    """Find random inputs with interesting content based on standard deviation"""
    idx_list = []
    std_list = []
    count = min(1000, len(dset))
    rand_idx_list = np.random.choice(len(dset), count, replace=False).tolist()
    for idx in rand_idx_list:
        inp = dset[idx][0]
        std_list.append(inp[0].std())
        idx_list.append(idx)
    # sort by std
    idx_list = np.array(idx_list)[np.argsort(std_list)][-40:]
    return idx_list.tolist()

def full_frame_evaluation(stitched_predictions, tar, inp):
    """Visualize full frame predictions with input and targets"""
    ncols = tar.shape[-1] + 1
    nrows = 2
    fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=(ncols * 5, nrows * 5))
    ax[0,0].imshow(inp)
    for i in range(ncols -1):
        vmin = stitched_predictions[...,i].min()
        vmax = stitched_predictions[...,i].max()
        ax[0,i+1].imshow(tar[...,i], vmin=vmin, vmax=vmax)
        ax[1,i+1].imshow(stitched_predictions[...,i], vmin=vmin, vmax=vmax)

    # Disable the axis for ax[1,0]
    ax[1,0].axis('off')
    ax[0,0].set_title("Input", fontsize=15)
    ax[0,1].set_title("Channel 1", fontsize=15)
    ax[0,2].set_title("Channel 2", fontsize=15)
    
    # Set y labels
    ax[0,2].yaxis.set_label_position("right")
    ax[0,2].set_ylabel("Target", fontsize=15)
    ax[1,2].yaxis.set_label_position("right")
    ax[1,2].set_ylabel("Predicted", fontsize=15)

def find_recent_metrics():
    """Find the most recent metrics file from lightning logs"""
    last_idx = 0
    fpath_schema = "./lightning_logs/version_{run_idx}/metrics.csv"
    assert os.path.exists(fpath_schema.format(run_idx=last_idx)), f"File {fpath_schema.format(run_idx=last_idx)} does not exist"
    while os.path.exists(fpath_schema.format(run_idx=last_idx)):
        last_idx += 1
    last_idx -= 1
    return fpath_schema.format(run_idx=last_idx)

def plot_metrics(df):
    """Plot training metrics from dataframe"""
    fig, ax = plt.subplots(figsize=(12,3), ncols=4)
    
    # Use proper column names that exist in your metrics file
    if 'reconstruction_loss_epoch' in df.columns:
        df['reconstruction_loss_epoch'].dropna().reset_index(drop=True).plot(ax=ax[0], marker='o')
    ax[0].set_title('Reconstruction Loss')
    ax[0].grid(True)
    
    if 'kl_loss_epoch' in df.columns:
        df['kl_loss_epoch'].dropna().reset_index(drop=True).plot(ax=ax[1], marker='o')
    ax[1].set_title('KL Divergence Loss')
    ax[1].grid(True)
    
    if 'val_loss' in df.columns:
        df['val_loss'].dropna().reset_index(drop=True).plot(ax=ax[2], marker='o')
    ax[2].set_title('Validation Loss')
    ax[2].grid(True)
    
    if 'val_psnr' in df.columns:
        df['val_psnr'].dropna().reset_index(drop=True).plot(ax=ax[3], marker='o')
    ax[3].set_title('Validation PSNR')
    ax[3].grid(True)
    
    # Set background color for all subplots
    for a in ax:
        a.set_facecolor('lightgray')
        a.set_xlabel("Epoch")
    
    plt.tight_layout()

def show_sampling(dset, model, ax=None):
    """Show different posterior samples for a given input"""
    idx_list = pick_random_inputs_with_content(dset)
    # inp, S1, S2, diff, mmse, tar
    ncols=6
    imgsz = 3
    if ax is None:
        _,ax = plt.subplots(figsize=(imgsz*ncols, imgsz*2), ncols=ncols, nrows=2)
    inp_patch, tar_patch = dset[idx_list[0]]
    ax[0,0].imshow(inp_patch[0])
    ax[0,0].set_title("Input (Idx: {})".format(idx_list[0]))

    samples = []
    n_samples = 50
    # get prediction 
    model.eval()
    for _ in range(n_samples):
        with torch.no_grad():
            pred_patch,_ = model(torch.Tensor(inp_patch).unsqueeze(0).to(model.device))
            samples.append(pred_patch[0,:tar_patch.shape[0]].cpu().numpy())
    samples = np.array(samples)

    ax[0,1].imshow(samples[0,0]); ax[0,1].set_title("Sample 1")
    ax[0,2].imshow(samples[1,0]); ax[0,2].set_title("Sample 2")
    ax[0,3].imshow(samples[0,0] - samples[1,0], cmap='coolwarm'); ax[0,3].set_title("S1 - S2")
    ax[0,4].imshow(np.mean(samples[:,0], axis=0)); ax[0,4].set_title("MMSE")
    ax[0,5].imshow(tar_patch[0]); ax[0,5].set_title("Target")
    # second channel
    ax[1,1].imshow(samples[0,1])
    ax[1,2].imshow(samples[1,1])
    ax[1,3].imshow(samples[0,1] - samples[1,1], cmap='coolwarm')
    ax[1,4].imshow(np.mean(samples[:,1], axis=0))
    ax[1,5].imshow(tar_patch[1])

    ax[1,0].axis('off')