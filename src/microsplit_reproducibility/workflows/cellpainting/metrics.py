"""
Image quality metrics for Cell Painting MicroSplit predictions.
"""

from typing import Dict, List, Optional

import numpy as np


DEFAULT_CHANNEL_NAMES = ["DNA", "RNA", "ER", "AGP", "Mito"]


def compute_metrics(
    channel_images: Dict[str, np.ndarray],
    mmse_prediction: np.ndarray,
    channel_names: Optional[List[str]] = None,
    data_range: float = 65535.0,
) -> Dict[str, float]:
    """Compute per-channel PSNR and SSIM.

    Parameters
    ----------
    channel_images : dict
        Mapping channel_name -> ndarray (H, W) ground-truth (uint16).
    mmse_prediction : ndarray (H, W, C)
        Denormalized MMSE prediction (float32).
    channel_names : list of str, optional
        Ordered channel names. Defaults to
        ``["DNA", "RNA", "ER", "AGP", "Mito"]``.
    data_range : float
        Dynamic range for PSNR/SSIM computation. Default: 65535.0 (uint16).

    Returns
    -------
    dict
        Keys ``psnr_{channel}`` and ``ssim_{channel}`` for each channel.
    """
    from skimage.metrics import peak_signal_noise_ratio, structural_similarity

    if channel_names is None:
        channel_names = DEFAULT_CHANNEL_NAMES

    metrics: Dict[str, float] = {}
    for ch_idx, ch_name in enumerate(channel_names):
        gt   = channel_images[ch_name].astype(np.float32)
        pred = np.clip(mmse_prediction[..., ch_idx], 0, data_range).astype(np.float32)
        metrics[f"psnr_{ch_name}"] = float(
            peak_signal_noise_ratio(gt, pred, data_range=data_range)
        )
        metrics[f"ssim_{ch_name}"] = float(
            structural_similarity(gt, pred, data_range=data_range)
        )
    return metrics
