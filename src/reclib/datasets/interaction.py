from typing import Any, Optional, Tuple

import numpy as np
import pandas as pd


class InteractionDataset(object):
    """Interaction dataset."""

    def __init__(  # noqa: C901
        self,
        df: Optional[pd.DataFrame] = None,
        df_csv_path: Optional[str] = None,
        user_ids: Optional[np.ndarray[Any, np.dtype[np.int64]]] = None,
        item_ids: Optional[np.ndarray[Any, np.dtype[np.int64]]] = None,
        user_id_col: str = "userId",
        item_id_col: str = "movieId",
        chrono_col: str = "timestamp",
    ):
        if df_csv_path is not None:
            self.interactions = pd.read_csv(df_csv_path)
        elif df is not None:
            self.interactions = df.copy()
        else:
            raise ValueError()

        self.user_col = user_id_col
        self.item_col = item_id_col
        self.chrono_col = chrono_col

        self.interactions.sort_values(by=self.chrono_col, inplace=True)
        self.user_group_by = self.interactions.groupby(by=self.user_col)
        self.user_groups = list(self.user_group_by.groups)

        if user_ids is None:
            user_ids = self.interactions[self.user_col].unique()
        if item_ids is None:
            item_ids = self.interactions[self.item_col].unique()

        if user_ids is not None:
            user_ids.sort()
        if item_ids is not None:
            item_ids.sort()

        self.user2index = {user: user_idx for user_idx, user in enumerate(user_ids)}
        self.index2user = {user_idx: user for user, user_idx in self.user2index.items()}
        self.interactions[self.user_col] = self.interactions[self.user_col].apply(
            self.user2index.get,
        )

        self.item2index = {item: item_idx + 2 for item_idx, item in enumerate(item_ids)}
        self.index2item = {item_idx: item for item, item_idx in self.item2index.items()}
        self.interactions[self.item_col] = self.interactions[self.item_col].apply(
            self.item2index.get,
        )

    @property
    def num_user(self) -> int:
        """
        Get number of users.

        :return: Number of users.
        """
        return len(self.user2index)

    @property
    def num_item(self) -> int:
        """
        Get number of items.

        :return: Number of item.
        """
        return len(self.item2index)

    def __len__(self) -> int:
        """
        Get number of interactions.

        :return: Number of interactions.
        """
        return len(self.interactions)

    def __str__(self) -> str:
        """
        Get info as string.

        :return: String.
        """
        return "INTERACTION DATA: {0} users, {1} items, {2} interactions".format(
            self.num_user,
            self.num_item,
            self.__len__(),
        )


def load_users_items(
    df_csv_path: str,
    user_id_col: str,
    item_id_col: str,
) -> Tuple[np.ndarray[Any, np.dtype[np.int64]], np.ndarray[Any, np.dtype[np.int64]]]:
    """
    Load interactions from csv dataframe.

    :return: Tuple of two numpy arrays.
    """
    df = pd.read_csv(df_csv_path)
    return df[user_id_col].unique(), df[item_id_col].unique()
