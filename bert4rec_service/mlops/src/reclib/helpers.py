import os
from typing import List

import numpy as np
import pandas as pd
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import DataLoader

from bert4rec_service.mlops import InteractionDataset
from bert4rec_service.mlops import SequentialItemsDataset
from src.reclib.models.sequential import BERT4Rec
from src.utils.db import DatabaseRepository as DBRepo


def get_handler(epochs, device, log_dir, checkpoint_dir, checkpoint_prefix):
    logger = TensorBoardLogger(
        save_dir=log_dir,
    )
    checkpoint_handler = ModelCheckpoint(
        monitor="val_loss",
        mode="min",
        dirpath=checkpoint_dir,
        filename=f"{checkpoint_prefix}_best",
    )
    return pl.Trainer(
        accelerator=device,
        max_epochs=epochs,
        logger=logger,
        callbacks=[checkpoint_handler],
        log_every_n_steps=1
    )


def get_dataloaders(
        splits,
        data_name,
        data_root,
        data_user_col,
        data_item_col,
        chrono_col,
        seq_length,
        batch_size,
        num_workers,
        mask_p=None,
):
    if isinstance(splits, str):
        splits = [splits]

    db_repo = DBRepo(os.path.join(data_root, f"{data_name}.db"))
    df = pd.DataFrame(
        db_repo.get_interactions([data_user_col, data_item_col, chrono_col]),
        columns=[data_user_col, data_item_col, chrono_col]
    )
    interactions_data = InteractionDataset(
        df=df,
        user_id_col=data_user_col,
        item_id_col=data_item_col,
        chrono_col=chrono_col
    )
    loaders = [
        DataLoader(
            SequentialItemsDataset(interactions_data, split, seq_length, mask_p),
            shuffle=split == 'train',
            batch_size=batch_size,
            num_workers=num_workers
        )
        for split in splits
    ]
    if len(loaders) == 1:
        return loaders[0]
    return loaders


class BERT4RecPredictor:
    def __init__(
            self,
            checkpoint_path,
            data_root,
            data_name,
            seq_length=120,
            data_user_col='userId',
            data_item_col='movieId',
            chrono_col='timestamp',
            device='cpu'
    ):
        db_repo = DBRepo(os.path.join(data_root, f"{data_name}.db"))
        df = pd.DataFrame(
            db_repo.get_interactions([data_user_col, data_item_col, chrono_col]),
            columns=[data_user_col, data_item_col, chrono_col]
        )
        self._interactions_data = InteractionDataset(
            df=df,
            user_id_col=data_user_col,
            item_id_col=data_item_col,
            chrono_col=chrono_col
        )
        self._seq_length = seq_length
        self._model = BERT4Rec.load_from_checkpoint(
            checkpoint_path=checkpoint_path,
            vocab_size=self._interactions_data.num_item + 2,
            mask_token=SequentialItemsDataset.mask_token,
            pad_token=SequentialItemsDataset.pad_token,
            map_location=device
        )
        self._model.eval()

    def predict(self, sequence: List, avoided_items: List = None):
        avoided_items = avoided_items if avoided_items else []
        avoided_items.extend(sequence)
        avoided_items.extend([SequentialItemsDataset.mask_token, SequentialItemsDataset.pad_token])
        try:
            sequence.remove(SequentialItemsDataset.pad_token)
        except ValueError:
            pass
        try:
            sequence.remove(SequentialItemsDataset.mask_token)
        except ValueError:
            pass
        if not len(sequence):
            return []
        sequence = SequentialItemsDataset.process_pre_infer(
            sequence,
            self._seq_length,
            self._interactions_data.item2index
        )
        ranked_items = np.vectorize(
            lambda it: self._interactions_data.index2item.get(it, SequentialItemsDataset.pad_token)
        )(
            self._model.predict_step(sequence.unsqueeze(0))[0].detach().cpu().numpy()
        )
        ranked_items = ranked_items[np.isin(ranked_items, avoided_items, invert=True)]
        return ranked_items.tolist()
