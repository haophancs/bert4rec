from typing import List

import numpy as np
import pandas as pd


class InteractionDataset(object):
    def __init__(self,
                 df: pd.DataFrame = None,
                 df_csv_path: str = None,
                 user_ids: np.array = None,
                 item_ids: np.array = None,
                 user_id_col: str = 'userId',
                 item_id_col: str = 'movieId',
                 chrono_col: str = 'timestamp'):
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

        user_ids.sort()
        item_ids.sort()

        self.user2index = {user: i for i, user in enumerate(user_ids)}
        self.index2user = {i: user for user, i in self.user2index.items()}
        self.interactions[self.user_col] = self.interactions[self.user_col].apply(self.user2index.get)

        self.item2index = {item: i + 2 for i, item in enumerate(item_ids)}
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


def load_users_items(df_csv_path: str, user_id_col, item_id_col):
    df = pd.read_csv(df_csv_path)
    return df[user_id_col].unique(), df[item_id_col].unique()
