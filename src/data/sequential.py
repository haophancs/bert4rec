import random
from copy import deepcopy

import pandas as pd
import numpy as np
import torch

from src.data.interaction import InteractionDataset


class SequentialItemsDataset(torch.utils.data.Dataset):
    def __init__(self,
                 interaction_data: InteractionDataset,
                 split: str,
                 seq_length=200,
                 mask_p=0.2):
        self.split = split
        self.interaction_data = deepcopy(interaction_data)
        self.pad_token = 0
        self.mask_token = 1
        self.seq_length = seq_length
        self.mask_p = mask_p

    def __len__(self):
        return len(self.interaction_data.user_groups)

    @property
    def vocab_size(self):
        return self.interaction_data.num_item + 2

    @staticmethod
    def get_sequence(interactions: pd.DataFrame, split: str, seq_length: int = 120, val_seq_length: int = 5):
        if split == "train":
            end_pos = random.randint(10, interactions.shape[0] - val_seq_length)
        elif split in ["val", "test"]:
            end_pos = interactions.shape[0]
        else:
            raise ValueError
        start_pos = max(0, end_pos - seq_length)
        sequence = interactions[start_pos:end_pos]
        return sequence

    def __getitem__(self, idx):
        user_group = self.interaction_data.user_groups[idx]
        interactions = self.interaction_data.user_group_by.get_group(user_group)
        actual_sequence = self.get_sequence(
            interactions, split=self.split, seq_length=self.seq_length
        )[self.interaction_data.item_col].values
        masked_sequence = actual_sequence.copy()
        if self.split == "train":
            masked_sequence = mask_array(masked_sequence, mask_val=self.mask_token, p=1 - self.mask_p)
        else:
            masked_sequence = mask_last_elements_array(masked_sequence, mask_val=self.mask_token)
        pad_mode = "left" if random.random() < 0.5 else "right"
        actual_sequence = pad_array(
            actual_sequence, length=self.seq_length, pad_val=self.pad_token, mode=pad_mode
        )
        masked_sequence = pad_array(
            masked_sequence, length=self.seq_length, pad_val=self.pad_token, mode=pad_mode
        )
        masked_sequence = torch.LongTensor(masked_sequence)
        actual_sequence = torch.LongTensor(actual_sequence)
        return masked_sequence, actual_sequence

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