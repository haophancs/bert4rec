import numpy as np
import torch
import torch.nn as nn

from src.data.sequential import pad_array
from src.metrics.sequential import masked_accuracy, masked_cross_entropy
from src.models.sequential.base import SequentialRecommender


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

        self.encoder_layers = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=hidden_size, nhead=num_attention_heads, dropout=dropout
            ), num_layers=num_hidden_layers
        )
        self.linear_layer = nn.Linear(hidden_size, vocab_size)

    def encode(self, item_ids):
        item_embeddings = self.item_embedding(item_ids)
        positional_embeddings = self.positional_embedding(
            torch.arange(0, item_embeddings.size(1)).to(self.device).unsqueeze(0).repeat(
                item_embeddings.size(0), 1
            )
        )
        embeddings = (item_embeddings + positional_embeddings).permute(1, 0, 2)
        hidden_states = self.encoder_layers(embeddings).permute(1, 0, 2)
        return hidden_states

    def forward(self, item_ids, *args):
        hidden_states = self.encode(item_ids)
        linear_output = self.linear_layer(hidden_states)
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

    def predict(self, item_ids, k=30):
        masked_sequence = torch.LongTensor(pad_array(
            np.array(item_ids + [self.mask_token]),
            length=self.seq_length,
            pad_val=self.pad_token,
            mode='left'
        )).unsqueeze(0)
        with torch.no_grad():
            prediction = self(masked_sequence)
        next_item_ids = prediction[0, -1].detach().cpu().numpy()
        next_item_ids = np.argsort(next_item_ids).tolist()[::-1][:k]
        next_item_ids = np.setdiff1d(next_item_ids, item_ids).tolist()
        return next_item_ids