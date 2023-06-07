import random
from copy import deepcopy

import pandas as pd
import torch

from src.data.utils import pad_array, mask_array, mask_last_elements_array


class InteractionDataset(object):
    def __init__(self,
                 data_csv_path,
                 user_col='userId',
                 item_col='movieId',
                 chrono_col='timestamp'):
        self.user_col = user_col
        self.item_col = item_col
        self.chrono_col = chrono_col

        self.interactions = pd.read_csv(data_csv_path)
        self.interactions.sort_values(by=self.chrono_col, inplace=True)

        self.user_group_by = self.interactions.groupby(by=self.user_col)
        self.user_groups = list(self.user_group_by.groups)

        self.user2index = {user: i for i, user in enumerate(self.interactions[self.user_col].unique())}
        self.index2user = {i: user for user, i in self.user2index.items()}
        self.interactions[self.user_col] = self.interactions[self.user_col].apply(self.user2index.get)

        self.item2index = {item: i + 2 for i, item in enumerate(self.interactions[self.item_col].unique())}
        self.index2item = {i: item for item, i in self.item2index.items()}
        self.interactions[self.item_col] = self.interactions[self.item_col].apply(self.item2index.get)

    @property
    def num_user(self):
        return len(self.user2index)

    @property
    def num_item(self):
        return len(self.item2index)

    def __len__(self):
        return len(self.interactions)

    def __str__(self):
        return f"INTERACTION DATA: {self.num_user} users, {self.num_item} items, {self.__len__()} interactions"


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
    def get_sequence(interactions: pd.DataFrame, split: str, seq_size: int = 120, val_seq_size: int = 5):
        if split == "train":
            end_pos = random.randint(10, interactions.shape[0] - val_seq_size)
        elif split in ["val", "test"]:
            end_pos = interactions.shape[0]
        else:
            raise ValueError
        start_pos = max(0, end_pos - seq_size)
        sequence = interactions[start_pos:end_pos]
        return sequence

    def __getitem__(self, idx):
        user_group = self.interaction_data.user_groups[idx]
        interactions = self.interaction_data.user_group_by.get_group(user_group)
        actual_sequence = self.get_sequence(
            interactions, split=self.split, seq_size=self.seq_length
        )[self.interaction_data.item_col].values
        masked_sequence = actual_sequence.copy()
        if self.split == "train":
            masked_sequence = mask_array(masked_sequence, mask_val=self.mask_token, p=1 - self.mask_p)
        else:
            masked_sequence = mask_last_elements_array(masked_sequence, mask_val=self.mask_token)
        pad_mode = "left" if random.random() < 0.5 else "right"
        actual_sequence = pad_array(
            actual_sequence, size=self.seq_length, pad_val=self.pad_token, mode=pad_mode
        )
        masked_sequence = pad_array(
            masked_sequence, size=self.seq_length, pad_val=self.pad_token, mode=pad_mode
        )
        masked_sequence = torch.LongTensor(masked_sequence)
        actual_sequence = torch.LongTensor(actual_sequence)
        return masked_sequence, actual_sequence
