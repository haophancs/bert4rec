import numpy as np


def mask_array(values: np.ndarray, mask_val: int, p=0.8):
    values = values.copy()
    mask = np.random.rand(len(values)) < p
    values[~mask] = mask_val
    return values


def mask_last_elements_array(values: np.ndarray, mask_val: int, val_seq_length: int = 5):
    values = values.copy()
    values[-val_seq_length:] = mask_array(values[-val_seq_length:], mask_val, p=0.5)
    return values


def pad_array(values: np.ndarray, length: int, pad_val: int, mode="left"):
    pad_length = length - len(values)
    if pad_length > 0:
        padding = np.full(pad_length, pad_val)
        if mode == "left":
            values = np.concatenate((padding, values))
        else:
            values = np.concatenate((values, padding))
    return values
