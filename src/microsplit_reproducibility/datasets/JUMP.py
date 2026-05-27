import os
import numpy as np
import tifffile
from pathlib import Path
from careamics.lvae_training.dataset import DataSplitType
from careamics.lvae_training.dataset.utils.data_utils import get_datasplit_tuples

def get_train_val_data(
    data_config,
    datadir,
    datasplit_type: DataSplitType,
    val_fraction=0.1,
    test_fraction=0.1,
    **kwargs,
):
    """
    Load and prepare JUMP dataset for training with robust file checking.
    """
    datadir = Path(datadir)
    # Get list of individual channel directories from config
    channel_list = data_config.channel_idx_list
    
    # 1. Load individual channel images
    channel_images = []
    expected_file_count = None

    for channel in channel_list:
        channel_dir = datadir / channel
        if not channel_dir.exists():
            raise FileNotFoundError(f"Directory not found: {channel_dir}")

        # Support both .tif and .tiff extensions
        files = sorted([
            f for f in os.listdir(channel_dir) 
            if f.lower().endswith('.tif') or f.lower().endswith('.tiff')
        ])
        
        if len(files) == 0:
            raise ValueError(f"No .tif or .tiff files found in {channel_dir}")
        
        # Sanity check: ensure all channels have the same number of images
        if expected_file_count is None:
            expected_file_count = len(files)
        elif len(files) != expected_file_count:
            raise ValueError(f"Consistency error: {channel} has {len(files)} files, "
                             f"but previous channel had {expected_file_count}")

        # Load all images for this channel
        images = [tifffile.imread(channel_dir / f) for f in files]
        
        # Stack images for this channel: [num_images, height, width]
        channel_images.append(np.stack(images))
    
    # 2. Load combined channel images
    combined_dir = datadir / 'combined'
    if not combined_dir.exists():
        raise FileNotFoundError(f"Required 'combined' directory not found in {datadir}")

    combined_files = sorted([
        f for f in os.listdir(combined_dir) 
        if f.lower().endswith('.tif') or f.lower().endswith('.tiff')
    ])
    
    if len(combined_files) == 0:
        raise ValueError(f"No .tif or .tiff files found in {combined_dir}")
    
    if len(combined_files) != expected_file_count:
        raise ValueError(f"Combined directory has {len(combined_files)} files, "
                         f"but channels have {expected_file_count}")

    combined_images = [tifffile.imread(combined_dir / f) for f in combined_files]
    combined_images = np.stack(combined_images)
    
    # 3. Format data for MicroSplit
    # Stack channels along the last axis: [num_images, height, width, num_channels]
    channel_stack = np.stack(channel_images, axis=-1)  
    
    # Concatenate combined images as the final channel: [num_images, height, width, num_channels+1]
    data = np.concatenate([channel_stack, combined_images[..., np.newaxis]], axis=-1)
    
    # 4. Split data into train, validation, and test sets
    train_idx, val_idx, test_idx = get_datasplit_tuples(
        val_fraction, test_fraction, len(data)
    )
    
    # Return the requested split
    if datasplit_type == DataSplitType.All:
        return data.astype(np.float32)
    elif datasplit_type == DataSplitType.Train:
        return data[train_idx].astype(np.float32)
    elif datasplit_type == DataSplitType.Val:
        return data[val_idx].astype(np.float32)
    elif datasplit_type == DataSplitType.Test:
        return data[test_idx].astype(np.float32)
    else:
        raise ValueError(f"Invalid datasplit type: {datasplit_type}")