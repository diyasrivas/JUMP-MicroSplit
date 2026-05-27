import os
from pathlib import Path
from typing import Callable, List, Union

import torch
from numpy.typing import NDArray

from careamics.lvae_training.dataset import DatasetConfig, DataType, DataSplitType
from careamics.lvae_training.dataset import (
    LCMultiChDloader,
    MultiChDloader,
    MultiFileDset,
)
try:
    from careamics.lvae_training.dataset import MultiChDloaderRef
except ImportError:
    MultiChDloaderRef = None
try:
    from careamics.lvae_training.dataset import MultiCropDset
except ImportError:
    MultiCropDset = None
from careamics.lvae_training.dataset.utils.data_utils import get_datasplit_tuples

from microsplit_reproducibility.datasets.lazy_dataset import LazyLCDataset

SplittingDataset = Union[LCMultiChDloader, MultiChDloader, MultiFileDset]


def create_train_val_datasets(
    datapath: str,
    train_config: DatasetConfig,
    val_config: DatasetConfig,
    test_config: DatasetConfig,
    load_data_func: Callable[..., NDArray],
) -> tuple[SplittingDataset, SplittingDataset, SplittingDataset, tuple[float, float]]:
    if train_config.data_type in [
        DataType.TavernaSox2Golgi,
        DataType.Dao3Channel,
        DataType.Dao3ChannelWithInput,
        # DataType.ExpMicroscopyV1,
        DataType.ExpMicroscopyV2,
        DataType.TavernaSox2GolgiV2,
    ]:
        dataset_class = MultiFileDset
    elif train_config.multiscale_lowres_count > 1:
        dataset_class = LCMultiChDloader
    elif train_config.data_type in [
        DataType.HTH23BData]:
        dataset_class = MultiChDloaderRef
    else:
        dataset_class = MultiChDloader

    train_data = dataset_class(
        train_config,
        datapath,
        load_data_fn=load_data_func,
        val_fraction=0.05,
        test_fraction=0.0,
    )
    max_val = train_data.get_max_val()
    val_config.max_val = max_val
    if train_config.datasplit_type == DataSplitType.All:
        val_config.datasplit_type = DataSplitType.All
        test_config.datasplit_type = DataSplitType.All # TODO temporary hack
    val_data = dataset_class(
        val_config,
        datapath,
        load_data_fn=load_data_func,
        val_fraction=0.05,
        test_fraction=0.0,
    )
    test_config.max_val = max_val
    test_data = dataset_class(
        test_config,
        datapath,
        load_data_fn=load_data_func,
        val_fraction=0.05,
        test_fraction=0.0,
    )
    mean_val, std_val = train_data.compute_mean_std()
    train_data.set_mean_std(mean_val, std_val)
    val_data.set_mean_std(mean_val, std_val)
    test_data.set_mean_std(mean_val, std_val)
    data_stats = train_data.get_mean_std()

    # NOTE: "input" mean & std are computed over the entire dataset and repeated for each channel.
    # On the contrary, "target" mean & std are computed separately for each channel.
    # manipulate data stats to only have one mean and std for the target
    assert isinstance(data_stats, tuple)
    assert isinstance(data_stats[0], dict)

    data_stats = (
        torch.tensor(data_stats[0]["target"]),
        torch.tensor(data_stats[1]["target"]),
    )

    return train_data, val_data, test_data, data_stats


def create_lazy_datasets(
    datapath: str,
    channel_names: List[str],
    train_grid_size: int = 64,
    val_grid_size: int = 32,
    image_size: int = 64,
    multiscale_lowres_count: int = 3,
    val_fraction: float = 0.1,
    test_fraction: float = 0.1,
    cache_size: int = 64,
    enable_rotation: bool = True,
) -> tuple[LazyLCDataset, LazyLCDataset, LazyLCDataset, tuple]:
    """Create train/val/test lazy-loading datasets from a JUMP-style directory.

    Discovers file paths once, splits frame indices, computes normalization
    stats via streaming (one pass through train frames), and creates three
    LazyLCDataset instances that share file references but differ in
    frame indices and augmentation settings.

    Parameters
    ----------
    datapath : str
        Path to the dataset directory containing channel subdirectories
        and a 'combined' subdirectory.
    channel_names : list[str]
        Ordered list of target channel names (e.g., ["DNA", "RNA", "ER", "AGP", "Mito"]).
    train_grid_size : int
        Grid size for training (controls patches per frame per epoch).
        Larger = fewer patches = faster epochs. Default 64.
    val_grid_size : int
        Grid size for validation/test (controls tiling coverage). Default 32.
    image_size : int
        Patch size (H=W). Default 64.
    multiscale_lowres_count : int
        Number of resolution scales. Default 3.
    val_fraction : float
        Fraction of data for validation. Default 0.1.
    test_fraction : float
        Fraction of data for test. Default 0.1.
    cache_size : int
        Number of frames in LRU cache. Default 64.
    enable_rotation : bool
        Whether to use rotation augmentation during training.

    Returns
    -------
    train_dset, val_dset, test_dset, data_stats
    """
    datapath = Path(datapath)

    # All channel names including combined
    all_channel_names = list(channel_names) + ["combined"]
    input_idx = len(channel_names)  # combined is the last channel
    target_idx_list = list(range(len(channel_names)))

    # Discover channel directories
    channel_dirs = {}
    for ch_name in all_channel_names:
        ch_dir = datapath / ch_name
        if not ch_dir.is_dir():
            raise FileNotFoundError(f"Channel directory not found: {ch_dir}")
        channel_dirs[ch_name] = ch_dir

    # Discover file list (from first channel, shared across all)
    ref_dir = channel_dirs[channel_names[0]]
    file_list = sorted([
        f for f in os.listdir(ref_dir)
        if f.lower().endswith('.tif') or f.lower().endswith('.tiff')
    ])
    if len(file_list) == 0:
        raise ValueError(f"No .tif files found in {ref_dir}")

    total_files = len(file_list)
    print(f"[LazyLC] Found {total_files} images in {datapath}")

    # Split into train/val/test indices
    train_idx, val_idx, test_idx = get_datasplit_tuples(
        val_fraction, test_fraction, total_files
    )
    print(f"[LazyLC] Split: train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)}")

    # Create train dataset and compute stats
    train_dset = LazyLCDataset(
        channel_dirs=channel_dirs,
        file_list=file_list,
        frame_indices=train_idx,
        image_size=image_size,
        grid_size=train_grid_size,
        multiscale_lowres_count=multiscale_lowres_count,
        input_idx=input_idx,
        target_idx_list=target_idx_list,
        channel_names=all_channel_names,
        enable_random_cropping=True,
        enable_rotation=enable_rotation,
        cache_size=cache_size,
    )

    # Compute stats via streaming (single pass through train frames)
    print("[LazyLC] Computing normalization statistics (streaming)...")
    mean_dict, std_dict = train_dset.compute_mean_std()
    max_val = train_dset.get_max_val()
    train_dset.set_mean_std(mean_dict, std_dict)
    print("[LazyLC] Stats computed.")

    # Create val dataset (deterministic tiling, no augmentation)
    val_dset = LazyLCDataset(
        channel_dirs=channel_dirs,
        file_list=file_list,
        frame_indices=val_idx,
        image_size=image_size,
        grid_size=val_grid_size,
        multiscale_lowres_count=multiscale_lowres_count,
        input_idx=input_idx,
        target_idx_list=target_idx_list,
        channel_names=all_channel_names,
        enable_random_cropping=False,
        enable_rotation=False,
        max_val=max_val,
        cache_size=cache_size,
    )
    val_dset.set_mean_std(mean_dict, std_dict)

    # Create test dataset (same as val)
    test_dset = LazyLCDataset(
        channel_dirs=channel_dirs,
        file_list=file_list,
        frame_indices=test_idx,
        image_size=image_size,
        grid_size=val_grid_size,
        multiscale_lowres_count=multiscale_lowres_count,
        input_idx=input_idx,
        target_idx_list=target_idx_list,
        channel_names=all_channel_names,
        enable_random_cropping=False,
        enable_rotation=False,
        max_val=max_val,
        cache_size=cache_size,
    )
    test_dset.set_mean_std(mean_dict, std_dict)

    # Build data_stats in the format expected by the training pipeline
    data_stats = (
        torch.tensor(mean_dict["target"]),
        torch.tensor(std_dict["target"]),
    )

    return train_dset, val_dset, test_dset, data_stats


def get_target_images(test_dset: SplittingDataset) -> NDArray:
    """Get the target images."""
    if test_dset.data_type in [
        DataType.HTIba1Ki67,
    ]:
        return test_dset._data
