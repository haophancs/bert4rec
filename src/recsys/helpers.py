import os

import numpy as np
import pandas as pd
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import DataLoader

from src.recsys.datasets.interaction import InteractionDataset
from src.recsys.datasets.sequential import SequentialItemsDataset
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
        user_ids=user_ids,
        item_ids=item_ids,
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
    return interactions_data, loaders
