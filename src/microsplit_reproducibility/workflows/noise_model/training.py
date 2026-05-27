"""
Noise model training utilities.

Wraps the Noise2Void → GaussianMixtureNoiseModel workflow used in all
Cell Painting experiments.  A single ``train_noise_model_for_channel``
function replaces the duplicated code in each experiment's
``2_noisemodels.py``.
"""

import gc
import os

import numpy as np
import torch
import tifffile
from careamics import CAREamist
from careamics.config import GaussianMixtureNMConfig, create_n2v_configuration
from careamics.models.lvae.noise_models import GaussianMixtureNoiseModel


def load_channel_data(dataset_dir: str, channel: str) -> np.ndarray:
    """Load all images for one channel into a (N, H, W, 1) float32 array.

    The extra trailing axis is required by CAREamics (SYXC format).

    Parameters
    ----------
    dataset_dir : str
        Path to the MicroSplit training dataset directory.
    channel : str
        Channel name (e.g., ``"DNA"``).

    Returns
    -------
    ndarray (N, H, W, 1) float32
    """
    channel_dir = os.path.join(dataset_dir, channel)
    if not os.path.isdir(channel_dir):
        raise FileNotFoundError(f"Channel directory not found: {channel_dir}")

    files = sorted(
        f for f in os.listdir(channel_dir) if f.endswith((".tif", ".tiff"))
    )
    if not files:
        raise FileNotFoundError(f"No TIFF files in {channel_dir}")

    images = [tifffile.imread(os.path.join(channel_dir, f)) for f in files]
    stack  = np.stack(images, axis=0)   # (N, H, W)
    return stack[..., np.newaxis]       # (N, H, W, 1)  — SYXC for CAREamics


def train_noise_model_for_channel(
    channel: str,
    dataset_dir: str,
    output_dir: str,
    n2v_epochs: int = 10,
    nm_epochs: int = 100,
    patch_size: int = 64,
    batch_size: int = 64,
    n_gaussian: int = 6,
    n_coeff: int = 4,
) -> None:
    """Train a per-channel Gaussian Mixture Noise Model via Noise2Void.

    Workflow:
      1. Load all channel images from ``dataset_dir/{channel}/``
      2. Train a Noise2Void model (N2V)
      3. Run N2V prediction to obtain denoised images
      4. Fit a GaussianMixtureNoiseModel on (signal, prediction) pairs
      5. Save the noise model as ``output_dir/noise_model_{channel}.npz``

    Parameters
    ----------
    channel : str
        Channel name (e.g., ``"DNA"``).
    dataset_dir : str
        Path to the MicroSplit training dataset directory.
    output_dir : str
        Directory in which to save the noise model .npz.
    n2v_epochs : int
        N2V training epochs. Default: 10.
    nm_epochs : int
        Noise model fitting epochs. Default: 100.
    patch_size : int
        N2V patch size. Default: 64.
    batch_size : int
        N2V batch size. Default: 64.
    n_gaussian : int
        Number of Gaussians in the mixture. Default: 6.
    n_coeff : int
        Number of polynomial coefficients. Default: 4.
    """
    print(f"\n{'=' * 60}")
    print(f"Processing channel: {channel}")
    print(f"{'=' * 60}")

    print("[1/6] Loading data...")
    input_data = load_channel_data(dataset_dir, channel)
    print(f"  Shape: {input_data.shape}, dtype: {input_data.dtype}")
    print(f"  Signal range: [{input_data.min():.1f}, {input_data.max():.1f}]")

    print("[2/6] Creating N2V configuration...")
    config = create_n2v_configuration(
        experiment_name=f"noise_model_n2v_{channel}",
        data_type="array",
        axes="SYXC",
        n_channels=1,
        patch_size=(patch_size, patch_size),
        batch_size=batch_size,
        num_epochs=n2v_epochs,
    )

    print("[3/6] Training N2V model...")
    work_dir  = os.path.join(output_dir, f"n2v_{channel}")
    careamist = CAREamist(source=config, work_dir=work_dir)
    careamist.train(train_source=input_data, val_minimum_split=5)

    print("[4/6] Running N2V prediction...")
    prediction = careamist.predict(input_data, tile_size=(256, 256))
    if isinstance(prediction, list):
        print(f"  {len(prediction)} prediction batches")

    print("[5/6] Post-processing...")
    channel_data       = input_data[..., 0]                        # (N, H, W)
    channel_prediction = np.concatenate(prediction)[:, 0]         # (N, H, W)
    print(
        f"  Signal    shape={channel_data.shape} "
        f"mean={channel_data.mean():.1f} std={channel_data.std():.1f}"
    )
    print(
        f"  Prediction shape={channel_prediction.shape} "
        f"mean={channel_prediction.mean():.1f} std={channel_prediction.std():.1f}"
    )

    del prediction
    torch.cuda.empty_cache()
    gc.collect()

    print("[6/6] Fitting GaussianMixture noise model...")
    nm_config = GaussianMixtureNMConfig(
        model_type="GaussianMixtureNoiseModel",
        min_signal=float(channel_data.min()),
        max_signal=float(channel_data.max()),
        n_coeff=n_coeff,
        n_gaussian=n_gaussian,
    )
    noise_model = GaussianMixtureNoiseModel(nm_config)
    noise_model.fit(
        signal=channel_data,
        observation=channel_prediction,
        n_epochs=nm_epochs,
    )

    os.makedirs(output_dir, exist_ok=True)
    noise_model.save(output_dir, f"noise_model_{channel}")
    print(f"  Saved: {output_dir}/noise_model_{channel}.npz")
    print(f"{'=' * 60}\n")
