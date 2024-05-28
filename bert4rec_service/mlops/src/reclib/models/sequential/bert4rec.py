import torch
import torch.nn as nn

from bert4rec_service.mlops import masked_accuracy, masked_cross_entropy
from src.reclib.models.sequential.base import SequentialRecommender


class BERT4Rec(SequentialRecommender):
    def __init__(
            self,
            *args,
            vocab_size,
            mask_token,
            pad_token,
            hidden_size=128,
            num_attention_heads=4,
            num_hidden_layers=6,
            dropout=0.4,
            **kwargs
    ):
        super(BERT4Rec, self).__init__(*args, **kwargs)
        self.vocab_size = vocab_size
        self.mask_token = mask_token
        self.pad_token = pad_token
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.num_hidden_layers = num_hidden_layers

        self.item_embedding = nn.Embedding(vocab_size, embedding_dim=hidden_size)
        self.positional_embedding = nn.Embedding(512, embedding_dim=hidden_size)

        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=hidden_size, nhead=num_attention_heads, dropout=dropout
            ), num_layers=num_hidden_layers
        )
        self.linear = nn.Linear(hidden_size, vocab_size)

    def encode(self, item_ids):
        item_embeddings = self.item_embedding(item_ids)
        positional_embeddings = self.positional_embedding(
            torch.arange(0, item_embeddings.size(1)).to(self.device).unsqueeze(0).repeat(
                item_embeddings.size(0), 1
            )
        )
        embeddings = (item_embeddings + positional_embeddings).permute(1, 0, 2)
        hidden_states = self.encoder(embeddings).permute(1, 0, 2)
        return hidden_states

    def forward(self, item_ids, *args):
        hidden_states = self.encode(item_ids)
        linear_output = self.linear(hidden_states)
        return linear_output

    def handle_batch(self, batch, inference=False):
        masked_sequence, actual_sequence = batch if not inference else (batch, None)
        predictions = self(masked_sequence)

        if inference:
            return predictions

        actual_sequence = actual_sequence.view(-1)
        masked_sequence = masked_sequence.view(-1)
        mask = masked_sequence == self.mask_token
        loss = masked_cross_entropy(predictions, truths=actual_sequence, mask=mask)
        accuracy = masked_accuracy(predictions, actual_sequence=actual_sequence, mask=mask)
        return predictions, loss, accuracy

    def batch_truths(self, batch):
        return batch[1][batch[0] == self.mask_token]
