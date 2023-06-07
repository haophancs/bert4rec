import numpy as np
import torch
import torch.nn.functional as F


def mask_array(values: np.ndarray, mask_val: int, p=0.8):
    values = values.copy()
    mask = np.random.rand(len(values)) < p
    values[~mask] = mask_val
    return values


def mask_last_elements_array(values: np.ndarray, mask_val: int, val_seq_size: int = 5):
    values = values.copy()
    values[-val_seq_size:] = mask_array(values[-val_seq_size:], mask_val, p=0.5)
    return values


def pad_array(values: np.ndarray, size: int, pad_val: int, mode="left"):
    pad_len = size - len(values)
    if pad_len > 0:
        padding = np.full(pad_len, pad_val)
        if mode == "left":
            values = np.concatenate((padding, values))
        else:
            values = np.concatenate((values, padding))
    return values
