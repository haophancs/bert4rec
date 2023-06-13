import random
from copy import deepcopy

import numpy as np
import pandas as pd
import torch

from src.reclib.datasets.interaction import InteractionDataset


class SequentialItemsDataset(torch.utils.data.Dataset):
    pad_token = 0
    mask_token = 1

    def __init__(self,
                 interaction_data: InteractionDataset,
                 split: str,
                 seq_length=200,
                 mask_p=0.2):
        self.split = split
        self.interaction_data = deepcopy(interaction_data)
        self.seq_length = seq_length
        self.mask_p = mask_p

    def __len__(self):
        return len(self.interaction_data.user_groups)

    @property
    def vocab_size(self):
        return self.interaction_data.num_item + 2

    @staticmethod
    def get_sequence(
            user_interacted: pd.DataFrame,
            split: str,
            seq_length: int = 120,
            val_seq_length: int = 5,
    ):
        if split == "train":
            end_pos = random.randint(10, user_interacted.shape[0] - val_seq_length)
        elif split in ["val", "test", "infer"]:
            end_pos = user_interacted.shape[0]
        else:
            raise ValueError
        start_pos = max(0, end_pos - seq_length)
        sequence = user_interacted[start_pos:end_pos]
        return sequence

    @staticmethod
    def process_pre_infer(sequence, seq_length, item2index=None):
        if item2index:
            sequence = [item2index[it] for it in sequence]
        sequence = np.array(sequence)
        masked_sequence = np.append(sequence, 1)[-seq_length:]
        masked_sequence = pad_array(masked_sequence, seq_length, SequentialItemsDataset.pad_token, mode='left')
        return torch.LongTensor(masked_sequence)

    def __getitem__(self, idx):
        user_group = self.interaction_data.user_groups[idx]
        user_interacted = self.interaction_data.user_group_by.get_group(user_group)
        current_sequence = self.get_sequence(
            user_interacted, split=self.split, seq_length=self.seq_length
        )[self.interaction_data.item_col].values
        masked_sequence = current_sequence.copy()

        if self.split == 'infer':
            return self.process_pre_infer(sequence=masked_sequence)

        if self.split == "train":
            masked_sequence = mask_array(masked_sequence, self.mask_token, p=self.mask_p)
        elif self.split == "val":
            masked_sequence = mask_last_elements_array(masked_sequence, self.mask_token, mask_length=5)
        else:
            masked_sequence = mask_last_elements_array(masked_sequence, self.mask_token, mask_length=1, p=1)

        pad_mode = "left" if self.split == 'test' or random.random() < 0.5 else "right"
        current_sequence = pad_array(current_sequence, self.seq_length, self.pad_token, pad_mode)
        masked_sequence = pad_array(masked_sequence, self.seq_length, self.pad_token, pad_mode)
        masked_sequence = torch.LongTensor(masked_sequence)
        current_sequence = torch.LongTensor(current_sequence)
        return masked_sequence, current_sequence


def mask_array(values: np.ndarray, mask_val: int, p=0.2):
    values = values.copy()
    mask = np.random.rand(len(values)) < (1 - p)
    values[~mask] = mask_val
    return values


def mask_last_elements_array(values: np.ndarray, mask_val: int, mask_length: int = 5, p=0.5):
    values = values.copy()
    values[-mask_length:] = mask_array(values[-mask_length:], mask_val, p=p)
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
