from pathlib import Path
from typing import List, Optional
import numpy as np
import matplotlib.pyplot as plt
import tifffile
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap


CHANNEL_COLORS = {
    "DNA": "blue",
    "RNA": "yellow",
    "ER": "green",
    "AGP": "orange",
    "Mito": "red"
}


def create_channel_colormap(channel_name: str) -> LinearSegmentedColormap:
    
    color = CHANNEL_COLORS.get(channel_name, "gray")
    
    if color == "yellow":
        colors = ["black", "yellow"]
    elif color == "blue":
        colors = ["black", "blue"]
    elif color == "green":
        colors = ["black", "green"]
    elif color == "orange":
        colors = ["black", "orange"]
    elif color == "red":
        colors = ["black", "red"]
    else:
        colors = ["black", "white"]
    
    return LinearSegmentedColormap.from_list(f"{channel_name}_cmap", colors)


def normalize_for_display(image: np.ndarray, percentile: float = 99.5) -> np.ndarray:
    
    vmin = np.percentile(image, 100 - percentile)
    vmax = np.percentile(image, percentile)
    
    normalized = (image - vmin) / (vmax - vmin)
    return np.clip(normalized, 0, 1)


def visualize_single_image(
    dataset_dir: Path,
    image_id: int,
    channels: List[str],
    figsize: tuple = (15, 5)
) -> plt.Figure:
    
    num_channels = len(channels)
    fig, axes = plt.subplots(1, num_channels + 2, figsize=figsize)
    
    combined_path = dataset_dir / "combined" / f"img_{image_id:05d}_combined.tif"
    combined = tifffile.imread(combined_path)
    combined_norm = normalize_for_display(combined)
    
    axes[0].imshow(combined_norm, cmap="gray")
    axes[0].set_title("Combined")
    axes[0].axis("off")
    
    for idx, channel in enumerate(channels, 1):
        channel_path = dataset_dir / channel / f"img_{image_id:05d}_{channel}.tif"
        channel_img = tifffile.imread(channel_path)
        channel_norm = normalize_for_display(channel_img)
        
        cmap = create_channel_colormap(channel)
        axes[idx].imshow(channel_norm, cmap=cmap)
        axes[idx].set_title(channel)
        axes[idx].axis("off")
    
    if len(channels) == 5:
        composite = create_5channel_composite(dataset_dir, image_id, channels)
        axes[-1].imshow(composite)
        axes[-1].set_title("5-Channel Composite")
        axes[-1].axis("off")
    
    plt.tight_layout()
    return fig


def create_5channel_composite(
    dataset_dir: Path,
    image_id: int,
    channels: List[str]
) -> np.ndarray:
    
    if len(channels) != 5:
        raise ValueError("Composite requires exactly 5 channels")
    
    channel_order = ["DNA", "Mito", "ER", "RNA", "AGP"]
    images = []
    
    for channel in channel_order:
        if channel not in channels:
            raise ValueError(f"Missing channel: {channel}")
        
        channel_path = dataset_dir / channel / f"img_{image_id:05d}_{channel}.tif"
        img = tifffile.imread(channel_path)
        images.append(normalize_for_display(img))
    
    composite = np.zeros((*images[0].shape, 3))
    
    composite[..., 2] = images[0]
    composite[..., 1] = 0.7 * images[2] + 0.3 * images[3]
    composite[..., 0] = 0.7 * images[1] + 0.3 * images[4]
    
    return np.clip(composite, 0, 1)


def visualize_dataset_sample(
    dataset_dir: Path,
    channels: List[str],
    num_samples: int = 4,
    seed: int = 42,
    figsize: tuple = (20, 5)
) -> plt.Figure:
    
    combined_dir = dataset_dir / "combined"
    all_images = sorted(combined_dir.glob("img_*.tif"))
    
    if len(all_images) == 0:
        raise ValueError(f"No images found in {combined_dir}")
    
    num_samples = min(num_samples, len(all_images))
    
    np.random.seed(seed)
    sampled_indices = np.random.choice(len(all_images), size=num_samples, replace=False)
    
    num_channels = len(channels)
    fig, axes = plt.subplots(num_samples, num_channels + 1, figsize=figsize)
    
    if num_samples == 1:
        axes = axes.reshape(1, -1)
    
    for row_idx, img_idx in enumerate(sampled_indices):
        image_id = int(all_images[img_idx].stem.split('_')[1])
        
        combined_path = all_images[img_idx]
        combined = tifffile.imread(combined_path)
        combined_norm = normalize_for_display(combined)
        
        axes[row_idx, 0].imshow(combined_norm, cmap="gray")
        axes[row_idx, 0].set_title(f"Combined (ID: {image_id})")
        axes[row_idx, 0].axis("off")
        
        for col_idx, channel in enumerate(channels, 1):
            channel_path = dataset_dir / channel / f"img_{image_id:05d}_{channel}.tif"
            channel_img = tifffile.imread(channel_path)
            channel_norm = normalize_for_display(channel_img)
            
            cmap = create_channel_colormap(channel)
            axes[row_idx, col_idx].imshow(channel_norm, cmap=cmap)
            axes[row_idx, col_idx].set_title(channel)
            axes[row_idx, col_idx].axis("off")
    
    plt.tight_layout()
    return fig


def verify_dataset_structure(
    dataset_dir: Path,
    channels: List[str]
) -> dict:
    
    results = {
        "valid": True,
        "issues": [],
        "statistics": {}
    }
    
    if not dataset_dir.exists():
        results["valid"] = False
        results["issues"].append(f"Dataset directory does not exist: {dataset_dir}")
        return results
    
    combined_dir = dataset_dir / "combined"
    if not combined_dir.exists():
        results["valid"] = False
        results["issues"].append("Combined directory missing")
        return results
    
    combined_images = list(combined_dir.glob("img_*.tif"))
    num_combined = len(combined_images)
    results["statistics"]["combined_images"] = num_combined
    
    if num_combined == 0:
        results["valid"] = False
        results["issues"].append("No combined images found")
        return results
    
    for channel in channels:
        channel_dir = dataset_dir / channel
        
        if not channel_dir.exists():
            results["valid"] = False
            results["issues"].append(f"Channel directory missing: {channel}")
            continue
        
        channel_images = list(channel_dir.glob(f"img_*_{channel}.tif"))
        num_channel = len(channel_images)
        results["statistics"][f"{channel}_images"] = num_channel
        
        if num_channel != num_combined:
            results["valid"] = False
            results["issues"].append(
                f"Channel {channel} has {num_channel} images, expected {num_combined}"
            )
    
    metadata_path = dataset_dir / "metadata.csv"
    if not metadata_path.exists():
        results["valid"] = False
        results["issues"].append("Metadata file missing")
    else:
        metadata = pd.read_csv(metadata_path)
        results["statistics"]["metadata_rows"] = len(metadata)
        
        if len(metadata) != num_combined:
            results["valid"] = False
            results["issues"].append(
                f"Metadata has {len(metadata)} rows, expected {num_combined}"
            )
    
    return results
