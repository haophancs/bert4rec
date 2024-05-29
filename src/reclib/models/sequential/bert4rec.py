from typing import Any, Tuple, Union

import torch
import torch.nn as nn  # noqa: WPS301

from src.reclib.metrics.sequential import masked_accuracy, masked_cross_entropy
from src.reclib.models.sequential.base import SequentialRecommender


class BERT4Rec(SequentialRecommender):
    """
    A BERT-based sequential recommender model.

    :param vocab_size: The size of the vocabulary.
    :param mask_token: The token used for masking.
    :param pad_token: The token used for padding.
    :param hidden_size: The size of the hidden layer. Default: 128.
    :param num_attention_heads: The number of attention heads. Default: 4.
    :param num_hidden_layers: The number of hidden layers. Default: 6.
    :param num_positional_embeddings: The number of positional embeddings. Default: 512.
    :param dropout: The dropout rate. Default: 0.4.
    """

    def __init__(
        self,
        *args: Any,
        vocab_size: int,
        mask_token: int,
        pad_token: int,
        hidden_size: int = 128,
        num_attention_heads: int = 4,
        num_hidden_layers: int = 6,
        num_positional_embeddings: int = 512,
        dropout: float = 0.4,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.vocab_size = vocab_size
        self.mask_token = mask_token
        self.pad_token = pad_token
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.num_hidden_layers = num_hidden_layers

        self.item_embedding = nn.Embedding(
            vocab_size,
            embedding_dim=hidden_size,
        )
        self.positional_embedding = nn.Embedding(
            num_positional_embeddings,
            embedding_dim=hidden_size,
        )

        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=hidden_size,
                nhead=num_attention_heads,
                dropout=dropout,
            ),
            num_layers=num_hidden_layers,
        )
        self.linear = nn.Linear(hidden_size, vocab_size)

    def encode(self, item_ids: torch.Tensor) -> torch.Tensor:
        """
        Encodes a sequence of items.

        :param item_ids: The input sequence of item IDs.
        :return: The encoded hidden states.
        """
        item_embeddings = self.item_embedding(item_ids)
        positional_embeddings = self.positional_embedding(
            torch.arange(0, item_embeddings.size(1))
            .to(self.device)
            .unsqueeze(
                0,
            )
            .repeat(
                item_embeddings.size(0),
                1,
            ),
        )
        embeddings = (item_embeddings + positional_embeddings).permute(1, 0, 2)
        return self.encoder(embeddings).permute(1, 0, 2)

    def forward(self, item_ids: torch.Tensor, *args: Any) -> torch.Tensor:
        """
        Forward item sequence to network.

        :param item_ids: The input sequence of item IDs.
        :param args: Additional arguments (optional).
        :return: The output of the model.
        """
        hidden_states = self.encode(item_ids)
        return self.linear(hidden_states)

    def handle_batch(
        self,
        batch: Any,
        inference: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """
        Handle batch of data.

        :param batch: The batch of data.
        :param inference: Whether to run in inference mode or not.
        :return: The predictions, loss, and accuracy.
        """
        if not inference:  # noqa: WPS504
            masked_sequence, actual_sequence = batch
        else:
            masked_sequence, actual_sequence = batch, None
        predictions = self(masked_sequence)

        if inference:
            return predictions

        actual_sequence = actual_sequence.view(-1)
        masked_sequence = masked_sequence.view(-1)
        mask = masked_sequence == self.mask_token
        loss = masked_cross_entropy(predictions, truths=actual_sequence, mask=mask)
        accuracy = masked_accuracy(
            predictions,
            actual_sequence=actual_sequence,
            mask=mask,
        )
        return predictions, loss, accuracy

    def batch_truths(self, batch: Any) -> torch.Tensor:
        """
        Get truths of each batch.

        :param batch: The batch of data.
        :return: The ground truth labels.
        """
        return batch[1][batch[0] == self.mask_token]
