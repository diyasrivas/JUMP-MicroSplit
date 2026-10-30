"""
Generic dataset builder for locally-downloaded Cell Painting plates.

Given a list of pre-selected FOVs, loads each FOV's channels, creates the
combined image, saves to a MicroSplit dataset directory, and returns a list of
metadata records.

The *sampling* strategy (which FOVs to select) is experiment-specific and lives
in each experiment's ``0_split.py`` or ``1_datasets.py``.  This module handles
the generic "process these FOVs" step that is identical across all experiments.
"""

import csv
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .image_io import (
    center_crop,
    combine_channels,
    fov_image_path,
    read_channel_paths,
    save_dataset_images,
)
from .manifest import PATH_PREFIX, TRAIN_ROLE, load_manifest, row_channel_paths


def build_dataset_from_paths(
    channel_paths: List[Dict[str, str]],
    output_dir: Path,
    channel_names: List[str],
    weights: Optional[List[float]] = None,
    normalize: bool = False,
    crop_size: Optional[int] = None,
    metadata: Optional[List[Dict]] = None,
) -> List[Dict]:
    """Build a MicroSplit training dataset from explicit per-channel image paths.

    For each FOV:
      1. Load each channel image from its path
      2. Optionally centre-crop every channel to ``crop_size``
      3. Create combined = weighted sum of all channels (default: equal weights)
      4. Save individual channels + combined to ``output_dir/``
      5. Record metadata

    Parameters
    ----------
    channel_paths : list of dict
        One mapping channel_name -> image path per FOV.
    output_dir : Path
        Where to write the dataset.
    channel_names : list of str
        Ordered channel names (e.g., ``["DNA", "RNA", "ER", "AGP", "Mito"]``).
    weights : list of float, optional
        Per-channel weights for the combined image.  Default: equal weights.
    normalize : bool
        If True, min-max normalise each channel before combining.
    crop_size : int, optional
        Centre-crop every image to ``crop_size`` x ``crop_size``. Needed when
        FOVs of different sizes go into one dataset.
    metadata : list of dict, optional
        Per-FOV metadata merged into each record. Must be the same length as
        ``channel_paths`` (or None).

    Returns
    -------
    list of dict
        One metadata record per successfully processed image.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    metadata_rows = []
    errors = 0

    for image_id, paths in enumerate(channel_paths):
        meta = metadata[image_id] if metadata is not None else {}
        try:
            channel_images = read_channel_paths({ch: paths[ch] for ch in channel_names})
            if crop_size is not None:
                channel_images = {
                    ch: center_crop(img, crop_size) for ch, img in channel_images.items()
                }

            combined = combine_channels(
                channel_images,
                channel_names,
                weights=weights,
                normalize=normalize,
            )

            save_dataset_images(combined, channel_images, image_id, output_dir, channel_names)

            record = {
                "image_id": image_id,
                **meta,
                **{f"{ch}_path": f"{ch}/{image_id:06d}.tiff" for ch in channel_names},
                "combined_path": f"combined/{image_id:06d}.tiff",
            }
            metadata_rows.append(record)

        except Exception as e:
            errors += 1
            if errors <= 10:
                print(f"  ERROR [{image_id}] {meta or paths}: {e}")
            continue

        if (image_id + 1) % 200 == 0:
            print(f"  {image_id + 1}/{len(channel_paths)} processed")

    if not metadata_rows:
        raise RuntimeError(
            f"Failed to process any images from {len(channel_paths)} FOVs. "
            "Check data paths and channel mapping."
        )

    meta_path = output_dir / "metadata.csv"
    with open(meta_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(metadata_rows[0].keys()))
        writer.writeheader()
        writer.writerows(metadata_rows)

    print(f"\nDataset written: {len(metadata_rows)} images (errors: {errors})")
    print(f"  Directory: {output_dir}")
    print(f"  metadata.csv: {meta_path}")

    return metadata_rows


def build_dataset_from_samples(
    samples: List[Tuple[str, str, str, int]],
    data_dir: Path,
    output_dir: Path,
    channel_names: List[str],
    channel_mapping: Dict[str, int],
    weights: Optional[List[float]] = None,
    normalize: bool = False,
    extra_metadata: Optional[List[Dict]] = None,
) -> List[Dict]:
    """Build a MicroSplit training dataset from pre-selected FOV locations.

    Images are read from ``data_dir/{batch}/images/{plate_dir}/Images/`` in the
    JUMP / Opera Phenix naming convention; see :func:`build_dataset_from_paths`
    for the processing steps.

    Parameters
    ----------
    samples : list of (batch, plate_dir, well, site)
        FOV locations to process.  ``plate_dir`` is the plate directory name
        (e.g., ``"BR00117015__2020-11-04T15_30_00-Measurement1"``).
    data_dir : Path
        Root of the locally downloaded dataset.
        Layout expected: ``data_dir/{batch}/images/{plate_dir}/Images/``
    output_dir : Path
        Where to write the dataset.
    channel_names : list of str
        Ordered channel names (e.g., ``["DNA", "RNA", "ER", "AGP", "Mito"]``).
    channel_mapping : dict
        Mapping channel_name -> channel file index
        (e.g., ``{"DNA": 5, "RNA": 3, "ER": 4, "AGP": 2, "Mito": 1}``).
    weights : list of float, optional
        Per-channel weights for the combined image.  Default: equal weights.
    normalize : bool
        If True, min-max normalise each channel before combining.
    extra_metadata : list of dict, optional
        Extra per-sample metadata to merge into each record. Must be the same
        length as ``samples`` (or None).

    Returns
    -------
    list of dict
        One metadata record per successfully processed image.
    """
    data_dir = Path(data_dir)
    channel_paths = []
    metadata = []
    for i, (batch, plate_dir, well, site) in enumerate(samples):
        images_dir = data_dir / batch / "images" / plate_dir / "Images"
        channel_paths.append({
            ch: str(fov_image_path(images_dir, well, site, channel_mapping[ch]))
            for ch in channel_names
        })
        record = {"batch": batch, "plate": plate_dir, "well": well, "site": site}
        if extra_metadata is not None:
            record.update(extra_metadata[i])
        metadata.append(record)

    return build_dataset_from_paths(
        channel_paths,
        output_dir,
        channel_names,
        weights=weights,
        normalize=normalize,
        metadata=metadata,
    )


def build_dataset_from_manifest(
    manifest_path: Path,
    cell_line: str,
    channel_names: List[str],
    output_dir: Path,
    crop_size: Optional[int] = None,
    max_fovs: int = 0,
) -> List[Dict]:
    """Build a MicroSplit training dataset from the training rows of a split manifest.

    The manifest must list one ``path_{channel}`` column per channel. The
    manifest is copied to ``output_dir/split_manifest_used.csv`` for provenance.

    Parameters
    ----------
    manifest_path : Path
        ``split_manifest.csv`` written by an experiment's ``0_split.py``.
    cell_line : str
        Only training rows of this cell line are used.
    channel_names : list of str
        Ordered channel names.
    output_dir : Path
        Where to write the dataset. Must not already contain a dataset.
    crop_size : int, optional
        Centre-crop every image to ``crop_size`` x ``crop_size``.
    max_fovs : int
        Use only the first ``max_fovs`` training rows (0 = all), for testing.

    Returns
    -------
    list of dict
        One metadata record per successfully processed image.

    Raises
    ------
    FileExistsError
        If ``output_dir`` already holds dataset images.
    ValueError
        If the manifest has no training rows for ``cell_line``.
    """
    manifest_path, output_dir = Path(manifest_path), Path(output_dir)
    combined_dir = output_dir / "combined"
    if combined_dir.is_dir() and any(combined_dir.iterdir()):
        raise FileExistsError(f"{output_dir} already contains a dataset")

    manifest = load_manifest(manifest_path)
    train = manifest[(manifest.role == TRAIN_ROLE) & (manifest.cell_line == cell_line)]
    if max_fovs > 0:
        train = train.head(max_fovs)
    if train.empty:
        raise ValueError(f"No training rows for {cell_line} in {manifest_path}")

    meta_columns = [c for c in train.columns if not c.startswith(PATH_PREFIX)]
    rows = train.to_dict("records")
    rows_out = build_dataset_from_paths(
        [row_channel_paths(r, channel_names) for r in rows],
        output_dir,
        channel_names,
        crop_size=crop_size,
        metadata=[{c: r[c] for c in meta_columns} for r in rows],
    )
    shutil.copy(manifest_path, output_dir / "split_manifest_used.csv")
    return rows_out
