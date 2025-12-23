from pathlib import Path
from typing import List, Tuple
import numpy as np
import tifffile


def save_prediction_outputs(
    predictions: np.ndarray,
    targets: np.ndarray,
    inputs: np.ndarray,
    channel_list: List[str],
    output_dir: Path,
    prefix: str = "test"
) -> None:
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    num_frames = predictions.shape[0]
    
    for frame_idx in range(num_frames):
        for ch_idx, channel in enumerate(channel_list):
            pred_file = output_dir / f"{prefix}_pred_frame{frame_idx}_{channel}.tif"
            target_file = output_dir / f"target_frame{frame_idx}_{channel}.tif"
            
            tifffile.imwrite(pred_file, predictions[frame_idx, 0, ch_idx])
            tifffile.imwrite(target_file, targets[frame_idx, 0, ch_idx])
        
        input_file = output_dir / f"input_frame{frame_idx}_combined.tif"
        tifffile.imwrite(input_file, inputs[frame_idx, 0])


def load_predictions_from_directory(
    prediction_dir: Path,
    channel_list: List[str],
    prefix: str = "test"
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    
    prediction_dir = Path(prediction_dir)
    
    pred_files = sorted(prediction_dir.glob(f"{prefix}_pred_frame*_{channel_list[0]}.tif"))
    num_frames = len(pred_files)
    
    if num_frames == 0:
        raise FileNotFoundError(f"No prediction files found in {prediction_dir}")
    
    sample_img = tifffile.imread(pred_files[0])
    height, width = sample_img.shape
    
    predictions = np.zeros((num_frames, 1, len(channel_list), height, width))
    targets = np.zeros((num_frames, 1, len(channel_list), height, width))
    inputs = np.zeros((num_frames, 1, height, width))
    
    for frame_idx in range(num_frames):
        for ch_idx, channel in enumerate(channel_list):
            pred_file = prediction_dir / f"{prefix}_pred_frame{frame_idx}_{channel}.tif"
            target_file = prediction_dir / f"target_frame{frame_idx}_{channel}.tif"
            
            predictions[frame_idx, 0, ch_idx] = tifffile.imread(pred_file)
            targets[frame_idx, 0, ch_idx] = tifffile.imread(target_file)
        
        input_file = prediction_dir / f"input_frame{frame_idx}_combined.tif"
        inputs[frame_idx, 0] = tifffile.imread(input_file)
    
    return predictions, targets, inputs


def verify_prediction_files(
    prediction_dir: Path,
    expected_frames: int,
    channel_list: List[str],
    prefix: str = "test"
) -> Tuple[List[str], List[str]]:
    
    prediction_dir = Path(prediction_dir)
    found = []
    missing = []
    
    for frame_idx in range(expected_frames):
        for channel in channel_list:
            pred_file = prediction_dir / f"{prefix}_pred_frame{frame_idx}_{channel}.tif"
            target_file = prediction_dir / f"target_frame{frame_idx}_{channel}.tif"
            
            if pred_file.exists():
                found.append(str(pred_file.name))
            else:
                missing.append(str(pred_file.name))
            
            if target_file.exists():
                found.append(str(target_file.name))
            else:
                missing.append(str(target_file.name))
        
        input_file = prediction_dir / f"input_frame{frame_idx}_combined.tif"
        if input_file.exists():
            found.append(str(input_file.name))
        else:
            missing.append(str(input_file.name))
    
    return found, missing
