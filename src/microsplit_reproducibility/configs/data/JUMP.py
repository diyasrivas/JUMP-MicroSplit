from typing import List

from careamics.lvae_training.dataset import DatasetConfig, DataSplitType, DataType

# Define available channels in the JUMP dataset
AVAILABLE_CHANNELS = ["DNA", "Mito", "RNA", "ER", "AGP"]

class JUMPDataConfig(DatasetConfig):
    """Configuration for JUMP dataset loading."""
    
    channel_idx_list: List[str]
    # List of channels to use (e.g., ["DNA", "Mito"])

def get_data_configs(
    channel_idx_list: List[str],
) -> tuple[JUMPDataConfig, JUMPDataConfig, JUMPDataConfig]:
    """Get the data configurations to use at training time.
    
    Parameters
    ----------
    channel_idx_list : List[str]
        The channels to use, should be a subset of AVAILABLE_CHANNELS.
        For example: ["DNA", "Mito"] or ["DNA", "ER"]
    
    Returns
    -------
    tuple[JUMPDataConfig, JUMPDataConfig, JUMPDataConfig]
        The train, validation and test data configurations.
    """
    # Validate channel selection
    for channel in channel_idx_list:
        if channel not in AVAILABLE_CHANNELS:
            raise ValueError(f"Channel {channel} not in available channels: {AVAILABLE_CHANNELS}")
    
    if len(channel_idx_list) < 2:
        raise ValueError("At least 2 channels must be specified for μSplit to work")
            
    train_data_config = JUMPDataConfig(
        data_type=DataType.SeparateTiffData,
        datasplit_type=DataSplitType.Train,
        image_size=[64, 64],
        grid_size=32,
        channel_idx_list=channel_idx_list,
        num_channels=len(channel_idx_list) + 1, 
        input_idx=len(channel_idx_list), 
        target_idx_list=list(range(len(channel_idx_list))), 
        multiscale_lowres_count=3,
        train_aug_rotate=True,
        target_separate_normalization=True,
        padding_kwargs={"mode": "reflect"},
        overlapping_padding_kwargs={"mode": "reflect"},
    )
    
    # Configuration for validation - disable augmentations
    val_data_config = train_data_config.model_copy(
        update=dict(
            datasplit_type=DataSplitType.Val,
            train_aug_rotate=False,  # No rotation augmentation for validation
            enable_random_cropping=False,  # No random cropping for validation
        )
    )
    
    # Configuration for testing - same as validation
    test_data_config = val_data_config.model_copy(
        update=dict(datasplit_type=DataSplitType.Test)
    )
    
    return train_data_config, val_data_config, test_data_config