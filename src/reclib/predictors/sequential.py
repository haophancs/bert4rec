import os
from typing import List, Optional

import numpy as np
import pandas as pd

from src.data.db import DatabaseRepository as DBRepo
from src.reclib.datasets.interaction import InteractionDataset
from src.reclib.datasets.sequential import SequentialItemsDataset
from src.reclib.models.sequential import BERT4Rec


class SequentialRecPredictor:
    model2class = {
        "bert4rec": BERT4Rec,
    }

    def __init__(
        self,
        checkpoint_path,
        data_root,
        data_name,
        seq_length=120,
        model_name="bert4rec",
        data_user_col="userId",
        data_item_col="movieId",
        chrono_col="timestamp",
        device="cpu",
    ):
        assert model_name in self.model2class.keys()
        print("Load data from database...")
        db_repo = DBRepo(os.path.join(data_root, f"{data_name}.db"))
        df = pd.DataFrame(
            db_repo.get_interactions([data_user_col, data_item_col, chrono_col]),
            columns=[data_user_col, data_item_col, chrono_col],
        )
        print("Prepare interaction data...")
        self._interactions_data = InteractionDataset(
            df=df,
            user_id_col=data_user_col,
            item_id_col=data_item_col,
            chrono_col=chrono_col,
        )
        self._seq_length = seq_length
        self._device = device
        print("Load model with pretrained weights...")
        self._model = self.model2class[model_name].load_from_checkpoint(
            checkpoint_path=checkpoint_path,
            vocab_size=self._interactions_data.num_item + 2,
            mask_token=SequentialItemsDataset.mask_token,
            pad_token=SequentialItemsDataset.pad_token,
            map_location=self._device,
        )
        self._model.eval()

    def predict(  # noqa: C901
        self,
        sequence: List[int],
        avoided_items: Optional[List[int]] = None,
    ) -> List[int]:
        if avoided_items is None:
            avoided_items = []
        avoided_items.extend(sequence)
        avoided_items.extend(
            [SequentialItemsDataset.mask_token, SequentialItemsDataset.pad_token],
        )
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
        sequence_tensor = (
            SequentialItemsDataset.process_pre_infer(
                sequence,
                self._seq_length,
                self._interactions_data.item2index,
            )
            .unsqueeze(0)
            .to(self._device)
        )

        ranked_items = np.vectorize(
            lambda it: self._interactions_data.index2item.get(
                it,
                SequentialItemsDataset.pad_token,
            ),
        )(
            self._model.predict_step(sequence_tensor)[0].detach().cpu().numpy(),
        )
        ranked_items = ranked_items[np.isin(ranked_items, avoided_items, invert=True)]
        return ranked_items.tolist()
