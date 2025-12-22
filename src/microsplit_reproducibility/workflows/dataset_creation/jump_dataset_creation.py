from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import tifffile
from dataclasses import dataclass
from enum import Enum


class JUMPDataset(Enum):
    PILOT = "cpg0000-jump-pilot"
    ORF = "cpg0016-jump"
    CRISPR = "cpg0016-jump"
    COMPOUND = "cpg0016-jump"


class Channel(Enum):
    DNA = "DNA"
    RNA = "RNA"
    ER = "ER"
    AGP = "AGP"
    MITO = "Mito"


@dataclass
class DatasetConfig:
    dataset_type: JUMPDataset
    channels: List[Channel]
    output_dir: Path
    num_images: int
    images_per_perturbation: int
    seed: int = 42
    normalize: bool = False
    weights: Optional[List[float]] = None
    source: str = "source_4"


@dataclass
class ImageMetadata:
    image_id: int
    perturbation_id: str
    perturbation_type: str
    source: str
    batch: str
    plate: str
    well: str
    site: int
    channels: List[str]
    gene_symbol: Optional[str] = None


def validate_channels(channels: List[Channel]) -> None:
    if len(channels) < 2:
        raise ValueError("At least 2 channels required for MicroSplit")
    
    if len(channels) != len(set(channels)):
        raise ValueError("Duplicate channels specified")


def validate_weights(weights: Optional[List[float]], num_channels: int) -> List[float]:
    if weights is None:
        return [1.0 / num_channels] * num_channels
    
    if len(weights) != num_channels:
        raise ValueError(f"Number of weights ({len(weights)}) must match number of channels ({num_channels})")
    
    if not np.isclose(sum(weights), 1.0):
        raise ValueError(f"Weights must sum to 1.0, got {sum(weights)}")
    
    return weights


def create_directory_structure(output_dir: Path, channels: List[Channel]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "combined").mkdir(exist_ok=True)
    
    for channel in channels:
        (output_dir / channel.value).mkdir(exist_ok=True)


def normalize_image(image: np.ndarray) -> np.ndarray:
    image_float = image.astype(np.float32)
    min_val = image_float.min()
    max_val = image_float.max()
    
    if max_val > min_val:
        return (image_float - min_val) / (max_val - min_val)
    return image_float


def combine_channels(
    channel_images: Dict[str, np.ndarray],
    channels: List[str],
    weights: List[float],
    normalize: bool = False
) -> Tuple[np.ndarray, Dict[str, float]]:
    
    if normalize:
        normalized_images = {
            ch: normalize_image(img) for ch, img in channel_images.items()
        }
    else:
        normalized_images = channel_images
    
    combined = np.zeros_like(list(normalized_images.values())[0], dtype=np.float32)
    
    for channel, weight in zip(channels, weights):
        combined += weight * normalized_images[channel]
    
    stats = {
        "mean": float(combined.mean()),
        "std": float(combined.std()),
        "min": float(combined.min()),
        "max": float(combined.max())
    }
    
    return combined, stats


def save_images(
    combined_image: np.ndarray,
    channel_images: Dict[str, np.ndarray],
    image_id: int,
    output_dir: Path,
    channels: List[str]
) -> Dict[str, Path]:
    
    paths = {}
    
    combined_filename = f"img_{image_id:05d}_combined.tif"
    combined_path = output_dir / "combined" / combined_filename
    tifffile.imwrite(combined_path, combined_image)
    paths["combined"] = combined_path
    
    for channel in channels:
        channel_filename = f"img_{image_id:05d}_{channel}.tif"
        channel_path = output_dir / channel / channel_filename
        tifffile.imwrite(channel_path, channel_images[channel])
        paths[channel] = channel_path
    
    return paths


def save_metadata(metadata_list: List[ImageMetadata], output_dir: Path) -> Path:
    records = []
    for metadata in metadata_list:
        record = {
            "image_id": metadata.image_id,
            "perturbation_id": metadata.perturbation_id,
            "perturbation_type": metadata.perturbation_type,
            "source": metadata.source,
            "batch": metadata.batch,
            "plate": metadata.plate,
            "well": metadata.well,
            "site": metadata.site,
            "combined_channels": "_".join(metadata.channels),
        }
        
        if metadata.gene_symbol:
            record["gene_symbol"] = metadata.gene_symbol
        
        records.append(record)
    
    df = pd.DataFrame(records)
    metadata_path = output_dir / "metadata.csv"
    df.to_csv(metadata_path, index=False)
    
    return metadata_path


def print_dataset_summary(
    metadata_list: List[ImageMetadata],
    output_dir: Path,
    channels: List[str]
) -> None:
    
    print(f"\nDataset created with {len(metadata_list)} images")
    print(f"- Combined images saved to: {output_dir / 'combined'}")
    
    for channel in channels:
        channel_count = sum(1 for _ in (output_dir / channel).glob("*.tif"))
        print(f"- {channel} channel: {channel_count} images saved to: {output_dir / channel}")
    
    print(f"- Metadata saved to: {output_dir / 'metadata.csv'}")
    print("\nDataset creation complete!")
