"""
Model and checkpoint loading utilities for Cell Painting MicroSplit experiments.
"""

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from careamics.lightning import VAEModule

from microsplit_reproducibility.configs.factory import (
    create_algorithm_config,
    get_likelihood_config,
    get_loss_config,
    get_model_config,
)
from microsplit_reproducibility.configs.parameters.JUMP import get_microsplit_parameters
from microsplit_reproducibility.notebook_utils.JUMP import load_pretrained_model


DEFAULT_CHANNEL_NAMES = ["DNA", "RNA", "ER", "AGP", "Mito"]


def find_checkpoint(training_dir: str) -> str:
    """Find the best (or most recent) checkpoint in training_dir/checkpoints/.

    Prefers checkpoints whose filename contains ``"best"``.

    Parameters
    ----------
    training_dir : str or Path
        Path to the training dataset directory (contains a ``checkpoints/``
        subdirectory).

    Returns
    -------
    str
        Absolute path to the chosen .ckpt file.

    Raises
    ------
    FileNotFoundError
        If no checkpoints directory or no .ckpt files are found.
    """
    ckpt_dir = os.path.join(training_dir, "checkpoints")
    if not os.path.isdir(ckpt_dir):
        raise FileNotFoundError(f"No checkpoints/ directory at {ckpt_dir}")
    ckpts = sorted(
        Path(ckpt_dir).glob("*.ckpt"),
        key=os.path.getmtime,
        reverse=True,
    )
    for ckpt in ckpts:
        if "best" in ckpt.stem.lower():
            return str(ckpt)
    if ckpts:
        return str(ckpts[0])
    raise FileNotFoundError(f"No .ckpt files found in {ckpt_dir}")


def load_model_and_stats(
    training_dir: str,
    checkpoint_path: str,
    channel_names: Optional[List[str]] = None,
) -> Tuple[VAEModule, Dict]:
    """Load a trained MicroSplit model and normalization statistics.

    Parameters
    ----------
    training_dir : str
        Path to the training dataset directory.  Must contain:
        - ``training_stats.npz``  (saved by the training script)
        - ``noise_models/``       (one .npz per channel)
        - ``checkpoints/``        (model weights)
    checkpoint_path : str
        Path to the .ckpt file to load.
    channel_names : list of str, optional
        Channel names used during training.
        Default: ``["DNA", "RNA", "ER", "AGP", "Mito"]``.

    Returns
    -------
    model : VAEModule
        Loaded and evaluated model on the appropriate device.
    stats : dict
        Keys: ``mean_input``, ``std_input``, ``mean_target``,
        ``std_target``, ``max_val``.
    """
    if channel_names is None:
        channel_names = DEFAULT_CHANNEL_NAMES

    noise_models_dir = os.path.join(training_dir, "noise_models")
    stats_path       = os.path.join(training_dir, "training_stats.npz")

    if not os.path.isfile(stats_path):
        raise FileNotFoundError(
            f"training_stats.npz not found at {stats_path}. "
            "Run the training script (3_train.py) first — it saves this file."
        )

    raw_stats = np.load(stats_path)
    stats = {k: raw_stats[k] for k in raw_stats.files}
    print(f"Loaded training stats from {stats_path}")

    experiment_params = get_microsplit_parameters(
        nm_path=noise_models_dir,
        channel_idx_list=channel_names,
    )
    experiment_params["data_stats"] = (
        torch.tensor(stats["mean_target"]),
        torch.tensor(stats["std_target"]),
    )

    model_config = get_model_config(**experiment_params)
    loss_config  = get_loss_config(**experiment_params)
    gaussian_lik_config, noise_model_config, nm_lik_config = get_likelihood_config(
        **experiment_params
    )
    experiment_config = create_algorithm_config(
        algorithm=experiment_params["algorithm"],
        loss_config=loss_config,
        model_config=model_config,
        gaussian_lik_config=gaussian_lik_config,
        nm_config=noise_model_config,
        nm_lik_config=nm_lik_config,
    )

    model = VAEModule(algorithm_config=experiment_config)
    load_pretrained_model(model, checkpoint_path)
    model.eval()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    print(f"Model loaded on {device}")

    return model, stats
