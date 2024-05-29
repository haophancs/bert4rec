import random
from copy import deepcopy
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch

from src.reclib.datasets.interaction import InteractionDataset


class SequentialItemsDataset(torch.utils.data.Dataset):  # type: ignore
    """
    A PyTorch dataset for sequential item recommendations.

    :param interaction_data: The InteractionDataset instance.
    :param split: The split type (train, val, test, or infer).
    :param seq_length: The maximum length of the sequence. Default: 200.
    :param mask_p: The probability of masking an item during training. Default: 0.2.
    """

    pad_token: int = 0
    mask_token: int = 1

    def __init__(
        self,
        interaction_data: InteractionDataset,
        split: str,
        seq_length: int = 200,
        mask_p: float = 0.2,
    ):
        self.split = split
        self.interaction_data = deepcopy(interaction_data)
        self.seq_length = seq_length
        self.mask_p = mask_p

    def __len__(self) -> int:
        return len(self.interaction_data.user_groups)

    @property
    def vocab_size(self) -> int:
        """
        Get size of the vocabulary.

        :return: The size of the vocabulary.
        """
        return self.interaction_data.num_item + 2

    @staticmethod
    def get_sequence(
        user_interacted: pd.DataFrame,
        split: str,
        seq_length: int = 120,
        val_seq_length: int = 5,
    ) -> pd.DataFrame:
        """
        Get the sequence of interactions for a user.

        :param user_interacted: DataFrame containing the user's interactions.
        :param split: The split type (train, val, test, or infer).
        :param seq_length: The maximum length of the sequence.
        :param val_seq_length: The length of the validation sequence.
        :raises ValueError: If the split type is not supported.
        :return: The sequence of interactions.
        """
        if split == "train":
            end_pos = random.randint(10, user_interacted.shape[0] - val_seq_length)
        elif split in {"val", "test", "infer"}:
            end_pos = user_interacted.shape[0]
        else:
            raise ValueError
        start_pos = max(0, end_pos - seq_length)
        return user_interacted[start_pos:end_pos]

    @staticmethod
    def process_pre_infer(
        sequence: List[int],
        seq_length: Optional[int] = None,
        item2index: Optional[Dict[int, int]] = None,
    ) -> torch.Tensor:
        """
        Preprocess the sequence for inference.

        :param sequence: The sequence of items.
        :param seq_length: The desired length of the sequence.
        :param item2index: A mapping from item IDs to indices.
        :return: The preprocessed sequence as a PyTorch tensor.
        """
        if item2index:
            sequence = [item2index[it] for it in sequence]
        sequence = np.array(sequence, dtype=np.int64)  # type: ignore
        if seq_length is None:
            seq_length = len(sequence)
        masked_sequence = np.append(sequence, 1)[-seq_length:]
        masked_sequence = pad_array(
            masked_sequence,
            seq_length,
            SequentialItemsDataset.pad_token,
            mode="left",
        )
        return torch.LongTensor(masked_sequence)

    def __getitem__(
        self,
        idx: int,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Get the masked and unmasked sequences for a user.

        :param idx: The index of the user group.
        :return: Tuple containing the masked and unmasked sequences as PyTorch tensors.
        """
        user_group = self.interaction_data.user_groups[idx]
        user_interacted = self.interaction_data.user_group_by.get_group(user_group)
        current_sequence = self.get_sequence(
            user_interacted,
            split=self.split,
            seq_length=self.seq_length,
        )[self.interaction_data.item_col].values.astype(np.int64)
        masked_sequence = current_sequence.copy()

        if self.split == "infer":
            return self.process_pre_infer(sequence=masked_sequence.tolist())

        if self.split == "train":
            masked_sequence = mask_array(
                masked_sequence,
                self.mask_token,
                mask_prob=self.mask_p,
            )
        elif self.split == "val":
            masked_sequence = mask_last_elements_array(
                masked_sequence,
                self.mask_token,
                mask_length=5,
            )
        else:
            masked_sequence = mask_last_elements_array(
                masked_sequence,
                self.mask_token,
                mask_length=1,
                mask_prob=1,
            )

        pad_mode = "left" if self.split == "test" or random.random() < 0.5 else "right"
        current_sequence = pad_array(
            current_sequence,
            self.seq_length,
            self.pad_token,
            pad_mode,
        )
        masked_sequence = pad_array(
            masked_sequence,
            self.seq_length,
            self.pad_token,
            pad_mode,
        )
        masked_sequence = torch.LongTensor(masked_sequence)
        current_sequence = torch.LongTensor(current_sequence)
        return masked_sequence, current_sequence


def mask_array(
    values: np.ndarray[Any, np.dtype[np.int64]],
    mask_val: int,
    mask_prob: float = 0.2,
) -> np.ndarray[Any, np.dtype[np.int64]]:
    """
    Mask elements in the input array with the given mask value.

    :param values: The input array.
    :param mask_val: The value to use for masking.
    :param mask_prob: The probability of masking an element.
    :return: The masked array.
    """
    values = values.copy()
    mask = np.random.rand(len(values)) < (1 - mask_prob)
    values[~mask] = mask_val
    return values


def mask_last_elements_array(
    values: np.ndarray[Any, np.dtype[np.int64]],
    mask_val: int,
    mask_length: int = 5,
    mask_prob: float = 0.5,
) -> np.ndarray[Any, np.dtype[np.int64]]:
    """
    Mask the last few elements in the input array with the given mask value.

    :param values: The input array.
    :param mask_val: The value to use for masking.
    :param mask_length: The number of elements to mask at the end of the array.
    :param mask_prob: The probability of masking an element.
    :return: The array with the last few elements masked.
    """
    values = values.copy()
    values[-mask_length:] = mask_array(
        values[-mask_length:],
        mask_val,
        mask_prob=mask_prob,
    )
    return values


def pad_array(
    values: np.ndarray[Any, np.dtype[np.int64]],
    length: int,
    pad_val: int,
    mode: str = "left",
) -> np.ndarray[Any, np.dtype[np.int64]]:
    """
    Pad the input array to the desired length.

    :param values: The input array.
    :param length: The desired length of the padded array.
    :param pad_val: The value to use for padding.
    :param mode: The mode of padding ('left' or 'right').
    :return: The padded array.
    """
    pad_length = length - len(values)
    if pad_length > 0:
        padding = np.full(pad_length, pad_val)
        if mode == "left":
            values = np.concatenate((padding, values))
        else:
            values = np.concatenate((values, padding))
    return values
