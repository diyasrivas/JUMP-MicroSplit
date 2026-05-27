from typing import List

from careamics.lvae_training.dataset import DatasetConfig, DataSplitType, DataType

# Default 5-channel order used across JUMP / Cell Painting Gallery experiments.
# This is the order in which channels are stacked and passed to the model.
DEFAULT_CHANNEL_NAMES = ["DNA", "RNA", "ER", "AGP", "Mito"]

# All channels available in JUMP Cell Painting data.
# A 6th channel can be added here if needed (e.g., "Actin" or "Brightfield").
AVAILABLE_CHANNELS = ["DNA", "RNA", "ER", "AGP", "Mito", "Actin", "Phase", "Brightfield"]


class JUMPDataConfig(DatasetConfig):
    """Configuration for JUMP / Cell Painting dataset loading."""

    channel_idx_list: List[str]
    """Ordered list of target channel names (e.g., ['DNA', 'RNA', 'ER', 'AGP', 'Mito'])."""


def get_data_configs(
    channel_idx_list: List[str] = None,
    train_grid_size: int = 32,
) -> tuple:
    """Return train, validation and test DatasetConfig objects.

    Parameters
    ----------
    channel_idx_list : list of str, optional
        Ordered target channel names.  Must have between 2 and 6 entries.
        Default: ``["DNA", "RNA", "ER", "AGP", "Mito"]``.
    train_grid_size : int
        Grid size for training patches.  Larger = fewer patches per epoch =
        faster training.  Default: 32.

    Returns
    -------
    tuple[JUMPDataConfig, JUMPDataConfig, JUMPDataConfig]
        (train_config, val_config, test_config)
    """
    if channel_idx_list is None:
        channel_idx_list = DEFAULT_CHANNEL_NAMES

    if len(channel_idx_list) < 2:
        raise ValueError(
            "At least 2 channels must be specified for μSplit to work."
        )
    if len(channel_idx_list) > 6:
        raise ValueError(
            f"At most 6 channels are supported; got {len(channel_idx_list)}."
        )

    train_data_config = JUMPDataConfig(
        data_type=DataType.SeparateTiffData,
        datasplit_type=DataSplitType.Train,
        image_size=[64, 64],
        grid_size=train_grid_size,
        channel_idx_list=channel_idx_list,
        num_channels=len(channel_idx_list) + 1,     # +1 for combined
        input_idx=len(channel_idx_list),             # combined is last
        target_idx_list=list(range(len(channel_idx_list))),
        multiscale_lowres_count=3,
        train_aug_rotate=True,
        target_separate_normalization=True,
        padding_kwargs={"mode": "reflect"},
        overlapping_padding_kwargs={"mode": "reflect"},
    )

    # Validation: no augmentations
    val_data_config = train_data_config.model_copy(
        update=dict(
            datasplit_type=DataSplitType.Val,
            train_aug_rotate=False,
            enable_random_cropping=False,
        )
    )

    # Test: same as validation
    test_data_config = val_data_config.model_copy(
        update=dict(datasplit_type=DataSplitType.Test)
    )

    return train_data_config, val_data_config, test_data_config
