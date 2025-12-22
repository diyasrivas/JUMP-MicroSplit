from pathlib import Path
from typing import List, Optional, Union
import numpy as np
import tifffile


def load_data_for_noise_model(
    dataset_dir: Union[str, Path],
    channels: List[str],
    max_images: Optional[int] = None
) -> np.ndarray:
    """
    Load channel data from dataset directory for noise model training.
    
    Parameters
    ----------
    dataset_dir : str or Path
        Path to dataset directory with channel subdirectories
    channels : list of str
        Channel names to load (e.g., ['DNA', 'RNA'])
    max_images : int, optional
        Maximum number of images to load per channel
        
    Returns
    -------
    np.ndarray
        Shape (N, H, W, C) where N=images, C=channels
    """
    dataset_dir = Path(dataset_dir)
    
    if not dataset_dir.exists():
        raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")
    
    channel_data = []
    for channel in channels:
        channel_dir = dataset_dir / channel
        if not channel_dir.exists():
            raise ValueError(f"Channel directory not found: {channel_dir}")
        
        tif_files = sorted(channel_dir.glob("*.tif"))
        if not tif_files:
            raise ValueError(f"No .tif files found in {channel_dir}")
        
        if max_images:
            tif_files = tif_files[:max_images]
        
        images = [tifffile.imread(str(f)) for f in tif_files]
        channel_data.append(np.stack(images))
    
    return np.stack(channel_data, axis=-1)
