"""
Cell Painting workflow utilities for MicroSplit.

Provides reusable components for all Cell Painting Gallery (CPG) experiments:
- image_io: FOV discovery, image loading, dataset saving
- dataset_builder: generic dataset building from locally downloaded plates
- noise_model: noise model training utilities
- prediction: tiling, stitching, prediction
- metrics: PSNR/SSIM computation
- model_io: checkpoint discovery and model loading
"""

from .image_io import (
    discover_fovs,
    read_fov_channels,
    get_available_sites,
    load_fov_image,
    combine_channels,
    save_dataset_images,
)

from .dataset_builder import build_dataset_from_samples

from .prediction import (
    generate_tile_positions,
    build_multiscale_frames,
    extract_patch_padded,
    build_input_patch,
    predict_fov,
    save_fov_predictions,
)

from .metrics import compute_metrics

from .model_io import find_checkpoint, load_model_and_stats

__all__ = [
    # image_io
    "discover_fovs",
    "read_fov_channels",
    "get_available_sites",
    "load_fov_image",
    "combine_channels",
    "save_dataset_images",
    # dataset_builder
    "build_dataset_from_samples",
    # prediction
    "generate_tile_positions",
    "build_multiscale_frames",
    "extract_patch_padded",
    "build_input_patch",
    "predict_fov",
    "save_fov_predictions",
    # metrics
    "compute_metrics",
    # model_io
    "find_checkpoint",
    "load_model_and_stats",
]
