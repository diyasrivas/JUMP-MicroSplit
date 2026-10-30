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
    read_channel_paths,
    get_available_sites,
    fov_image_path,
    load_fov_image,
    center_crop,
    combine_channels,
    save_dataset_images,
)

from .dataset_builder import (
    build_dataset_from_paths,
    build_dataset_from_samples,
    build_dataset_from_manifest,
)

from .prediction import (
    generate_tile_positions,
    build_multiscale_frames,
    extract_patch_padded,
    build_input_patch,
    predict_fov,
    save_fov_predictions,
    predict_manifest_fovs,
)

from .metrics import compute_metrics

from .model_io import find_checkpoint, load_model_and_stats

from .manifest import (
    TRAIN_ROLE,
    TEST_ROLES,
    channel_path_column,
    row_channel_paths,
    load_manifest,
    manifest_rows_for_plate,
    append_metadata_row,
    completed_fovs,
    write_cellprofiler_load_data,
)

__all__ = [
    # image_io
    "discover_fovs",
    "read_fov_channels",
    "read_channel_paths",
    "get_available_sites",
    "fov_image_path",
    "load_fov_image",
    "center_crop",
    "combine_channels",
    "save_dataset_images",
    # dataset_builder
    "build_dataset_from_paths",
    "build_dataset_from_samples",
    "build_dataset_from_manifest",
    # prediction
    "generate_tile_positions",
    "build_multiscale_frames",
    "extract_patch_padded",
    "build_input_patch",
    "predict_fov",
    "save_fov_predictions",
    "predict_manifest_fovs",
    # metrics
    "compute_metrics",
    # model_io
    "find_checkpoint",
    "load_model_and_stats",
    # manifest
    "TRAIN_ROLE",
    "TEST_ROLES",
    "channel_path_column",
    "row_channel_paths",
    "load_manifest",
    "manifest_rows_for_plate",
    "append_metadata_row",
    "completed_fovs",
    "write_cellprofiler_load_data",
]
