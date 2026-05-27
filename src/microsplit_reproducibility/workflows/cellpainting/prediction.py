"""
MicroSplit plate-level prediction utilities for Cell Painting data.

Provides tiling, stitching, inference and saving functions that are shared
across all Cell Painting experiments.  The only experiment-specific input is
the ``channel_names`` list (and the model/stats objects, which are loaded by
:mod:`model_io`).
"""

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import tifffile
from skimage.transform import resize


DEFAULT_CHANNEL_NAMES = ["DNA", "RNA", "ER", "AGP", "Mito"]


# ---------------------------------------------------------------------------
# Tiling helpers
# ---------------------------------------------------------------------------

def generate_tile_positions(
    H: int,
    W: int,
    image_size: int,
    grid_size: int,
) -> Tuple[List[int], List[int]]:
    """Generate tile start positions using ShiftBoundary logic.

    Parameters
    ----------
    H, W : int
        Full image dimensions.
    image_size : int
        Patch height/width.
    grid_size : int
        Stride between patches.

    Returns
    -------
    h_positions, w_positions : list of int
        Start row/column indices.
    """
    def _positions(total: int, img_sz: int, grd_sz: int) -> List[int]:
        pos = []
        p = 0
        while p + img_sz <= total:
            pos.append(p)
            p += grd_sz
        # Ensure the last tile reaches the end
        if not pos or pos[-1] + img_sz < total:
            pos.append(max(0, total - img_sz))
        return pos

    return _positions(H, image_size, grid_size), _positions(W, image_size, grid_size)


def build_multiscale_frames(
    combined: np.ndarray,
    multiscale_lowres_count: int,
) -> List[np.ndarray]:
    """Build progressively downsampled versions of the combined image.

    Parameters
    ----------
    combined : ndarray (H, W)
        Full-resolution combined image.
    multiscale_lowres_count : int
        Total number of scale levels (1 = no downsampling).

    Returns
    -------
    list of ndarray
        ``[full_res, half_res, quarter_res, ...]``, length
        ``multiscale_lowres_count``.
    """
    frames = [combined.astype(np.float32)]
    current = combined.astype(np.float32)
    for _ in range(1, multiscale_lowres_count):
        h, w = current.shape
        current = resize(
            current, (h // 2, w // 2),
            preserve_range=True, anti_aliasing=True,
        ).astype(np.float32)
        frames.append(current)
    return frames


def extract_patch_padded(
    data_2d: np.ndarray,
    h_start: int,
    w_start: int,
    size: int,
) -> np.ndarray:
    """Extract a square patch with reflect-padding for out-of-bounds regions."""
    H, W = data_2d.shape
    h_end, w_end = h_start + size, w_start + size

    if h_start >= 0 and w_start >= 0 and h_end <= H and w_end <= W:
        return data_2d[h_start:h_end, w_start:w_end].astype(np.float32)

    vh_s, vh_e = max(0, h_start), min(H, h_end)
    vw_s, vw_e = max(0, w_start), min(W, w_end)
    patch = data_2d[vh_s:vh_e, vw_s:vw_e]

    pad_t = vh_s - h_start
    pad_b = h_end - vh_e
    pad_l = vw_s - w_start
    pad_r = w_end - vw_e
    if pad_t > 0 or pad_b > 0 or pad_l > 0 or pad_r > 0:
        patch = np.pad(patch, ((pad_t, pad_b), (pad_l, pad_r)), mode="reflect")
    return patch.astype(np.float32)


def build_input_patch(
    multiscale_frames: List[np.ndarray],
    h_start: int,
    w_start: int,
    image_size: int,
    multiscale_lowres_count: int,
) -> np.ndarray:
    """Build the multi-scale input tensor for one tile position.

    Returns
    -------
    ndarray of shape (multiscale_lowres_count, image_size, image_size).
    """
    patches = [
        multiscale_frames[0][
            h_start : h_start + image_size,
            w_start : w_start + image_size,
        ][np.newaxis]
    ]
    h_center = h_start + image_size // 2
    w_center = w_start + image_size // 2

    for scale in range(1, multiscale_lowres_count):
        h_center = h_center // 2
        w_center = w_center // 2
        hs = h_center - image_size // 2
        ws = w_center - image_size // 2
        scaled_patch = extract_patch_padded(
            multiscale_frames[scale], hs, ws, image_size
        )
        patches.append(scaled_patch[np.newaxis])

    return np.concatenate(patches, axis=0)  # (scales, H, W)


# ---------------------------------------------------------------------------
# FOV prediction
# ---------------------------------------------------------------------------

def predict_fov(
    model,
    channel_images: Dict[str, np.ndarray],
    stats: Dict,
    channel_names: Optional[List[str]] = None,
    image_size: int = 64,
    grid_size: int = 32,
    multiscale_lowres_count: int = 3,
    mmse_count: int = 50,
    num_posterior_samples: int = 3,
    posterior_seeds: Tuple[int, ...] = (42, 123, 456),
    batch_size: int = 32,
) -> Tuple[np.ndarray, List[np.ndarray]]:
    """Run MicroSplit prediction on a single field of view.

    Parameters
    ----------
    model : VAEModule
        Trained model (already on the appropriate device).
    channel_images : dict
        Mapping channel_name -> ndarray (H, W), raw uint16 pixel values.
    stats : dict
        Normalization statistics as saved by the training script
        (keys: ``mean_input``, ``std_input``, ``mean_target``,
        ``std_target``, ``max_val``).
    channel_names : list of str, optional
        Ordered channel names. Defaults to
        ``["DNA", "RNA", "ER", "AGP", "Mito"]``.
    image_size : int
        Patch size (H=W). Default: 64.
    grid_size : int
        Stride between patches. Default: 32.
    multiscale_lowres_count : int
        Number of resolution scales. Default: 3.
    mmse_count : int
        Number of Monte-Carlo samples for MMSE. Default: 50.
    num_posterior_samples : int
        How many individual posterior samples to return. Default: 3.
    posterior_seeds : tuple of int
        RNG seeds for each saved posterior sample.
    batch_size : int
        Tile batch size for GPU inference. Default: 32.

    Returns
    -------
    mmse_prediction : ndarray (H, W, C) float32, denormalized
    posterior_samples : list of ndarray (H, W, C) float32, denormalized
    """
    if channel_names is None:
        channel_names = DEFAULT_CHANNEL_NAMES

    device = next(model.parameters()).device
    max_val    = stats["max_val"]
    mean_input = float(stats["mean_input"].mean())
    std_input  = float(stats["std_input"].mean())
    mean_target = stats["mean_target"].squeeze()
    std_target  = stats["std_target"].squeeze()
    if mean_target.ndim > 1:
        mean_target = mean_target.reshape(-1)
        std_target  = std_target.reshape(-1)

    H, W = next(iter(channel_images.values())).shape
    num_channels = len(channel_names)

    # Stack and clip channels
    channel_stack = np.stack(
        [channel_images[ch].astype(np.float32) for ch in channel_names], axis=-1
    )  # (H, W, C)
    for ch_idx in range(num_channels):
        channel_stack[..., ch_idx] = np.minimum(
            channel_stack[..., ch_idx], max_val[ch_idx]
        )

    # Combined input = sum of all channels
    combined = channel_stack.sum(axis=-1).astype(np.float32)
    combined  = np.minimum(combined, max_val[-1])

    ms_frames  = build_multiscale_frames(combined, multiscale_lowres_count)
    h_positions, w_positions = generate_tile_positions(H, W, image_size, grid_size)
    n_tiles    = len(h_positions) * len(w_positions)

    # Build all input patches
    all_patches = np.empty(
        (n_tiles, multiscale_lowres_count, image_size, image_size),
        dtype=np.float32,
    )
    tile_idx = 0
    for h in h_positions:
        for w in w_positions:
            all_patches[tile_idx] = build_input_patch(
                ms_frames, h, w, image_size, multiscale_lowres_count
            )
            tile_idx += 1
    all_patches = (all_patches - mean_input) / std_input

    mmse_accum = np.zeros(
        (n_tiles, num_channels, image_size, image_size), dtype=np.float64
    )
    posterior_tile_samples = [
        np.zeros((n_tiles, num_channels, image_size, image_size), dtype=np.float32)
        for _ in range(num_posterior_samples)
    ]

    model.eval()
    with torch.no_grad():
        for sample_idx in range(mmse_count):
            is_saved_sample = sample_idx < num_posterior_samples
            if is_saved_sample:
                torch.manual_seed(posterior_seeds[sample_idx])
                if torch.cuda.is_available():
                    torch.cuda.manual_seed(posterior_seeds[sample_idx])

            tiles_out = []
            for bs in range(0, n_tiles, batch_size):
                be  = min(bs + batch_size, n_tiles)
                inp = torch.from_numpy(all_patches[bs:be]).to(device)
                rec, _ = model(inp)
                if model.model.predict_logvar is not None:
                    rec, _ = torch.chunk(rec, 2, dim=1)
                tiles_out.append(rec[:, :num_channels].cpu().numpy())

            tiles_out = np.concatenate(tiles_out, axis=0)
            mmse_accum += tiles_out
            if is_saved_sample:
                posterior_tile_samples[sample_idx] = tiles_out.astype(np.float32)

    mmse_tiles = (mmse_accum / mmse_count).astype(np.float32)

    # -----------------------------------------------------------------------
    # Stitching
    # -----------------------------------------------------------------------
    def stitch(tile_predictions: np.ndarray) -> np.ndarray:
        output  = np.zeros((H, W, num_channels), dtype=np.float32)
        overlap = (image_size - grid_size) // 2
        ti = 0
        for h in h_positions:
            for w in w_positions:
                pred   = tile_predictions[ti]
                h_from = overlap if h > 0 else 0
                w_from = overlap if w > 0 else 0
                h_to   = image_size - overlap if (h + image_size < H) else image_size
                w_to   = image_size - overlap if (w + image_size < W) else image_size
                for ch in range(num_channels):
                    output[
                        h + h_from : h + h_to,
                        w + w_from : w + w_to,
                        ch,
                    ] = pred[ch, h_from:h_to, w_from:w_to]
                ti += 1
        return output

    def denormalize(img: np.ndarray) -> np.ndarray:
        for ch in range(num_channels):
            img[..., ch] = img[..., ch] * std_target[ch] + mean_target[ch]
        return img

    mmse_stitched        = denormalize(stitch(mmse_tiles))
    posterior_stitched   = [
        denormalize(stitch(pt)) for pt in posterior_tile_samples
    ]

    return mmse_stitched, posterior_stitched


# ---------------------------------------------------------------------------
# Output saving
# ---------------------------------------------------------------------------

def save_fov_predictions(
    output_dir: str,
    well: str,
    site: int,
    mmse_prediction: np.ndarray,
    posterior_samples: List[np.ndarray],
    channel_names: Optional[List[str]] = None,
    save_uint16: bool = True,
) -> None:
    """Save MMSE + posterior sample TIFFs for one FOV.

    Output structure::

        output_dir/
            mmse/    mmse_{well}_s{site:02d}_{channel}.tif
            sample_0/ sample0_{well}_s{site:02d}_{channel}.tif
            sample_1/ ...

    Parameters
    ----------
    output_dir : str or Path
    well : str
    site : int
    mmse_prediction : ndarray (H, W, C) float32
    posterior_samples : list of ndarray (H, W, C) float32
    channel_names : list of str, optional
    save_uint16 : bool
        If True (default) clip and cast to uint16; otherwise save float32.
    """
    if channel_names is None:
        channel_names = DEFAULT_CHANNEL_NAMES

    # MMSE
    mmse_dir = os.path.join(output_dir, "mmse")
    os.makedirs(mmse_dir, exist_ok=True)
    for ch_idx, ch_name in enumerate(channel_names):
        img = mmse_prediction[..., ch_idx]
        if save_uint16:
            img = np.clip(img, 0, 65535).astype(np.uint16)
        tifffile.imwrite(
            os.path.join(mmse_dir, f"mmse_{well}_s{site:02d}_{ch_name}.tif"),
            img,
        )

    # Posterior samples
    for s_idx, sample in enumerate(posterior_samples):
        sdir = os.path.join(output_dir, f"sample_{s_idx}")
        os.makedirs(sdir, exist_ok=True)
        for ch_idx, ch_name in enumerate(channel_names):
            img = sample[..., ch_idx]
            if save_uint16:
                img = np.clip(img, 0, 65535).astype(np.uint16)
            tifffile.imwrite(
                os.path.join(sdir, f"sample{s_idx}_{well}_s{site:02d}_{ch_name}.tif"),
                img,
            )
