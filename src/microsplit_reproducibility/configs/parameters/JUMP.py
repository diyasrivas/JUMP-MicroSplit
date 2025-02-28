import os
from typing import Literal, List

from ._base import SplittingParameters

def get_musplit_parameters(
    channel_idx_list: list[str], 
) -> dict:
    return SplittingParameters(
        algorithm="musplit",
        img_size=(64, 64),
        target_channels=len(channel_idx_list),
        multiscale_count=3,
        predict_logvar="pixelwise",
        loss_type="musplit",
        kl_type="kl_restricted",
    ).model_dump()

def _get_nm_paths(
    nm_path: str, 
    channel_idx_list: list[str], 
) -> list[str]:
    nm_paths = []
    for channel in channel_idx_list:
        fname = f"noise_model_{channel}.npz" 
        nm_paths.append(os.path.join(nm_path, fname))
    return nm_paths

def get_microsplit_parameters(
    nm_path: str,
    channel_idx_list: List[str],
    batch_size: int = 32,
) -> dict:
    nm_paths = _get_nm_paths(nm_path=nm_path, channel_idx_list=channel_idx_list)
    return SplittingParameters(
        algorithm="denoisplit",
        img_size=(64, 64),
        target_channels=len(channel_idx_list), 
        multiscale_count=3,
        predict_logvar="pixelwise",
        loss_type="denoisplit_musplit",
        nm_paths=nm_paths,
        kl_type="kl_restricted",
        batch_size=batch_size,
    ).model_dump()

def get_eval_params() -> dict:
    raise NotImplementedError("Evaluation parameters not implemented for JUMP.")
    return SplittingParameters().model_dump()