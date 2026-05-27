"""
Cell Painting image I/O utilities.

Handles reading raw plate images from the standard JUMP / Cell Painting Gallery
file naming convention:
    r{row:02d}c{col:02d}f{site:02d}p{plane:02d}-ch{ch_idx}sk1fk1fl1.tiff

Also provides helpers for building training dataset directories (channel
subdirectories + combined/).
"""

import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import tifffile


# ---------------------------------------------------------------------------
# FOV discovery
# ---------------------------------------------------------------------------

def discover_fovs(
    plate_images_dir: str,
) -> List[Tuple[Tuple[str, int], str]]:
    """Discover all unique FOVs (well, site) in a plate Images/ directory.

    Scans for files matching the JUMP Cell Painting naming convention and
    returns one entry per (well, site) combination.

    Parameters
    ----------
    plate_images_dir : str or Path
        Path to the Images/ subdirectory of a plate.

    Returns
    -------
    list of ((well_str, site_int), filename_prefix)
        Sorted by (well, site).
        ``filename_prefix`` is like ``"r01c02f03p01"`` (without the
        ``-ch{N}sk1fk1fl1.tiff`` suffix).
    """
    pattern = re.compile(
        r"r(\d{2})c(\d{2})f(\d{2})p(\d{2})-ch(\d+)sk1fk1fl1\.tiff?"
    )
    fovs: Dict[Tuple[str, int], str] = {}
    for fname in os.listdir(plate_images_dir):
        m = pattern.match(fname)
        if not m:
            continue
        row, col, field, plane = (
            int(m.group(1)),
            int(m.group(2)),
            int(m.group(3)),
            int(m.group(4)),
        )
        well = f"{chr(64 + row)}{col:02d}"
        key = (well, field)
        if key not in fovs:
            fovs[key] = f"r{row:02d}c{col:02d}f{field:02d}p{plane:02d}"
    return sorted(fovs.items(), key=lambda x: x[0])


def get_available_sites(images_dir: Path) -> List[Tuple[str, int]]:
    """Return sorted (well, site) pairs from filenames in an Images/ directory.

    Parameters
    ----------
    images_dir : Path
        Path to the Images/ subdirectory of a plate.

    Returns
    -------
    list of (well_str, site_int), sorted.
    """
    sites: set = set()
    for p in Path(images_dir).glob("*.tiff"):
        stem = p.stem.split("-")[0]  # e.g. "r01c02f03p01"
        row  = int(stem[1:3])
        col  = int(stem[4:6])
        site = int(stem[7:9])
        well = f"{chr(64 + row)}{col:02d}"
        sites.add((well, site))
    return sorted(sites)


# ---------------------------------------------------------------------------
# Image reading
# ---------------------------------------------------------------------------

def read_fov_channels(
    plate_images_dir: str,
    filename_prefix: str,
    channel_mapping: Dict[str, int],
) -> Dict[str, np.ndarray]:
    """Read channel images for a single FOV.

    Parameters
    ----------
    plate_images_dir : str or Path
    filename_prefix : str
        Like ``"r01c02f03p01"`` (returned by :func:`discover_fovs`).
    channel_mapping : dict
        Mapping from channel name (e.g., ``"DNA"``) to channel file index
        (e.g., ``5``).

    Returns
    -------
    dict mapping channel_name -> ndarray (H, W) uint16
    """
    channels: Dict[str, np.ndarray] = {}
    for ch_name, ch_idx in channel_mapping.items():
        path = os.path.join(
            plate_images_dir,
            f"{filename_prefix}-ch{ch_idx}sk1fk1fl1.tiff",
        )
        channels[ch_name] = tifffile.imread(path)
    return channels


def load_fov_image(
    images_dir: Path,
    well: str,
    site: int,
    channel: str,
    channel_mapping: Dict[str, int],
) -> np.ndarray:
    """Load a single channel image for a specific well/site.

    Parameters
    ----------
    images_dir : Path
        Path to the Images/ directory of the plate.
    well : str
        Well identifier like ``"A01"`` or ``"P24"``.
    site : int
        Site/field number (1-indexed).
    channel : str
        Channel name like ``"DNA"``.
    channel_mapping : dict
        Mapping from channel name to file channel index.

    Returns
    -------
    ndarray (H, W) uint16
    """
    row = ord(well[0]) - 64
    col = int(well[1:])
    ch_idx = channel_mapping[channel]
    fname = f"r{row:02d}c{col:02d}f{site:02d}p01-ch{ch_idx}sk1fk1fl1.tiff"
    return tifffile.imread(str(Path(images_dir) / fname))


# ---------------------------------------------------------------------------
# Channel combining
# ---------------------------------------------------------------------------

def combine_channels(
    channel_images: Dict[str, np.ndarray],
    channel_names: List[str],
    weights: Optional[List[float]] = None,
    normalize: bool = False,
) -> np.ndarray:
    """Combine multiple channels into a single image via weighted sum.

    Parameters
    ----------
    channel_images : dict
        Mapping channel_name -> ndarray (H, W).
    channel_names : list of str
        Ordered list of channels to combine. Must all be in ``channel_images``.
    weights : list of float, optional
        Per-channel weights. Must sum to 1 (or be None for equal weighting).
        If None, equal weights (1/N each) are used.
    normalize : bool
        If True, min-max normalise each channel to [0, 1] before combining.

    Returns
    -------
    combined : ndarray (H, W) float32
    """
    if weights is None:
        weights = [1.0 / len(channel_names)] * len(channel_names)

    if len(weights) != len(channel_names):
        raise ValueError(
            f"len(weights)={len(weights)} must match "
            f"len(channel_names)={len(channel_names)}"
        )

    combined = None
    for ch, w in zip(channel_names, weights):
        img = channel_images[ch].astype(np.float32)
        if normalize:
            lo, hi = img.min(), img.max()
            if hi > lo:
                img = (img - lo) / (hi - lo)
        if combined is None:
            combined = w * img
        else:
            combined += w * img

    return combined.astype(np.float32)


# ---------------------------------------------------------------------------
# Dataset directory writing
# ---------------------------------------------------------------------------

def save_dataset_images(
    combined: np.ndarray,
    channel_images: Dict[str, np.ndarray],
    image_id: int,
    output_dir: Path,
    channel_names: List[str],
) -> None:
    """Save individual channel images + combined image to dataset directory.

    Output structure::

        output_dir/
            combined/{image_id:06d}.tiff   (float32)
            DNA/{image_id:06d}.tiff        (original dtype)
            RNA/{image_id:06d}.tiff
            ...

    Parameters
    ----------
    combined : ndarray (H, W)
        Pre-computed combined/sum image.
    channel_images : dict
        Mapping channel_name -> ndarray (H, W).
    image_id : int
        Sequential image index.
    output_dir : Path
        Root of the dataset directory.
    channel_names : list of str
        Which channels to save (must be keys in ``channel_images``).
    """
    output_dir = Path(output_dir)
    combined_dir = output_dir / "combined"
    combined_dir.mkdir(parents=True, exist_ok=True)
    tifffile.imwrite(
        str(combined_dir / f"{image_id:06d}.tiff"),
        combined.astype(np.float32),
    )
    for ch_name in channel_names:
        ch_dir = output_dir / ch_name
        ch_dir.mkdir(parents=True, exist_ok=True)
        tifffile.imwrite(
            str(ch_dir / f"{image_id:06d}.tiff"),
            channel_images[ch_name],
        )
