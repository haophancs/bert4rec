import pandas as pd


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
