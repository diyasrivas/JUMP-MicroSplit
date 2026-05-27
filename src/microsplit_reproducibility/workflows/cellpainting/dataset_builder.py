"""
Generic dataset builder for locally-downloaded Cell Painting plates.

Given a list of pre-selected (batch, plate, well, site) locations, loads each
FOV's channels, creates the combined image, saves to a MicroSplit dataset
directory, and returns a list of metadata records.

The *sampling* strategy (which FOVs to select) is experiment-specific and lives
in each experiment's ``1_datasets.py``.  This module handles the generic
"process these FOVs" step that is identical across all experiments.
"""

import csv
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from .image_io import load_fov_image, combine_channels, save_dataset_images


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

    For each (batch, plate_dir, well, site) sample:
      1. Load each channel image from ``data_dir/{batch}/images/{plate_dir}/Images/``
      2. Create combined = weighted sum of all channels (default: equal weights)
      3. Save individual channels + combined to ``output_dir/``
      4. Record metadata

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
    data_dir   = Path(data_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    metadata_rows = []
    errors = 0

    for image_id, (batch, plate_dir, well, site) in enumerate(samples):
        images_dir = data_dir / batch / "images" / plate_dir / "Images"

        try:
            channel_images = {
                ch: load_fov_image(images_dir, well, site, ch, channel_mapping)
                for ch in channel_names
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
                "batch": batch,
                "plate": plate_dir,
                "well": well,
                "site": site,
                **{f"{ch}_path": f"{ch}/{image_id:06d}.tiff" for ch in channel_names},
                "combined_path": f"combined/{image_id:06d}.tiff",
            }
            if extra_metadata is not None:
                record.update(extra_metadata[image_id])
            metadata_rows.append(record)

        except Exception as e:
            errors += 1
            if errors <= 10:
                print(f"  ERROR [{image_id}] {batch}/{plate_dir}/{well} s{site}: {e}")
            continue

        if (image_id + 1) % 200 == 0:
            print(f"  {image_id + 1}/{len(samples)} processed")

    if not metadata_rows:
        raise RuntimeError(
            f"Failed to process any images from {len(samples)} samples. "
            "Check data paths and channel mapping."
        )

    # Write metadata.csv
    meta_path = output_dir / "metadata.csv"
    with open(meta_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(metadata_rows[0].keys()))
        writer.writeheader()
        writer.writerows(metadata_rows)

    print(f"\nDataset written: {len(metadata_rows)} images (errors: {errors})")
    print(f"  Directory: {output_dir}")
    print(f"  metadata.csv: {meta_path}")

    return metadata_rows
