import os

import numpy as np

from careamics.lvae_training.dataset import DataSplitType
from skimage.io import imread


def load_tiff(path):
    """
    Returns a 4d numpy array: num_imgs*h*w*num_channels
    """
    data = imread(path, plugin="tifffile")
    return data


def load_tiff_test(path, idx):
    """
    Returns a 4d numpy array: num_imgs*h*w*num_channels
    """
    data = imread(path, plugin="tifffile")[idx, ..., 1] # we only want the 2nd channel
    return data


def _pick_subset(weight_arr, skip_idxs, allocated_weight, thresh):
    output_idxs = []
    remaining_w = allocated_weight
    for idx in range(len(weight_arr)):
        if idx in skip_idxs:
            continue
        if remaining_w < 0:
            break
        if remaining_w >= weight_arr[idx] - thresh:
            output_idxs.append(idx)
            remaining_w -= weight_arr[idx]
    return output_idxs


def _load_train_val_data(
    datadir, datasplit_type, val_fraction, test_fraction, thresh=0.01, train_test_split=True
):
    fnames = np.random.RandomState(955).permutation(sorted(os.listdir(datadir)))

    data = [load_tiff(os.path.join(datadir, fname)) for fname in fnames]
    if datasplit_type == DataSplitType.All:
        return data

    size_list = np.array([np.prod(d.shape) for d in data])
    size_list = size_list / size_list.sum()

    val_idx = _pick_subset(size_list, [], val_fraction, thresh)
    test_idx = _pick_subset(size_list, val_idx, test_fraction, thresh)
    train_idx = [i for i in range(len(data)) if i not in val_idx + test_idx]

    # train_idx = train_idx + val_idx + test_idx
    # val_idx = train_idx
    # test_idx = train_idx
    
    print(f"Train: {len(train_idx)} Val: {len(val_idx)} Test: {len(test_idx)}")
    # print(size_list[train_idx].sum(), size_list[val_idx].sum(), size_list[test_idx].sum())
    if datasplit_type == DataSplitType.Train:
        data = [data[i] for i in train_idx]
    elif datasplit_type == DataSplitType.Val:
        data = [data[i] for i in val_idx]
    elif datasplit_type == DataSplitType.Test:
        data = [data[i] for i in test_idx]

    return data


def get_train_val_data(
    data_config,
    datadir,
    datasplit_type: DataSplitType,
    val_fraction=None,
    test_fraction=None,
    **kwargs,
):
    if data_config.channel_list:
        data_arr = []
        for channel in data_config.channel_list:
            data = _load_train_val_data(
                os.path.join(datadir, channel), datasplit_type, val_fraction, test_fraction
            )
            data_arr.append(data)
    else:
        data_arr = _load_train_val_data(
            datadir, datasplit_type, val_fraction, test_fraction
        )
    return data_arr

def get_test_data(
    data_config,
    datadir,
    datasplit_type: DataSplitType,
    val_fraction=None,
    test_fraction=None,
    **kwargs,
):
    data = [load_tiff_test(os.path.join(datadir, fname), data_config.test_frame_idx) for fname in os.listdir(datadir)]
    print(f"Test: {len(data)}")
    # returning tuple because train dataset expects more than 1 channel
    return (data,)
