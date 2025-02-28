import os
import numpy as np
import tifffile
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
    """Load and prepare JUMP dataset for training.
    
    Parameters
    ----------
    data_config : JUMPDataConfig
        Configuration specifying which channels to load.
    datadir : str
        Base directory containing the prepared dataset.
    datasplit_type : DataSplitType
        Which split of data to load (train, val, test).
    val_fraction : float
        Fraction of data to use for validation.
    test_fraction : float
        Fraction of data to use for testing.
    
    Returns
    -------
    numpy.ndarray
        Loaded and prepared data with shape [num_images, height, width, num_channels+1]
        where the last dimension contains individual channels followed by the combined channel.
    """
    # Get list of individual channel directories
    channel_list = data_config.channel_idx_list
    
    # Load individual channel images
    channel_images = []
    for channel in channel_list:
        channel_dir = os.path.join(datadir, channel)
        files = sorted([f for f in os.listdir(channel_dir) if f.endswith('.tif')])
        
        # Load all images for this channel
        images = []
        for file in files:
            img = tifffile.imread(os.path.join(channel_dir, file))
            images.append(img)
        
        # Stack all images for this channel
        channel_images.append(np.stack(images))
    
    # Load combined channel images
    combined_dir = os.path.join(datadir, 'combined')
    combined_files = sorted([f for f in os.listdir(combined_dir) if f.endswith('.tif')])
    combined_images = []
    for file in combined_files:
        img = tifffile.imread(os.path.join(combined_dir, file))
        combined_images.append(img)
    combined_images = np.stack(combined_images)
    
    # Stack all channels along the last axis
    # Format: [num_images, height, width, num_channels+1]
    # The last channel is the combined channel
    channel_stack = np.stack(channel_images, axis=-1)  # [num_images, height, width, num_channels]
    data = np.concatenate([channel_stack, combined_images[..., np.newaxis]], axis=-1)
    
    # Split data into train, validation, and test sets
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