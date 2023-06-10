import numpy as np
import torch
import pytorch_lightning as pl


class SequentialRecommender(pl.LightningModule):
    def __init__(
            self,
            seq_length=120,
            lr=None,
            weight_decay=None,
            optimizer=None,
            scheduler=None,
            *args,
            **kwargs
    ):
        super(SequentialRecommender, self).__init__()
        self.lr = lr
        self.seq_length = seq_length
        self.weight_decay = weight_decay
        self.optimizer = optimizer
        self.scheduler = scheduler

    def forward(self, item_ids, *args):
        raise NotImplementedError()

    def handle_batch(self, batch, inference=False):
        raise NotImplementedError()

    def training_step(self, batch, *args):
        _, loss, accuracy = self.handle_batch(batch)
        self.log("train_loss", loss)
        self.log("train_accuracy", accuracy)
        return loss

    def validation_step(self, batch, *args):
        _, loss, accuracy = self.handle_batch(batch)
        self.log("val_loss", loss)
        self.log("val_accuracy", accuracy)
        return loss

    def test_step(self, batch, *args):
        predictions, loss, accuracy = self.handle_batch(batch, inference=False)
        self.log("test_loss", loss)
        self.log("test_accuracy", accuracy)
        return loss

    def predict_step(self, batch, *args, k=10):
        with torch.no_grad():
            predictions = self.handle_batch(batch, inference=True)
        next_item_ids = predictions[-1].detach().cpu().numpy()
        next_item_ids = np.argsort(next_item_ids).tolist()[::-1][:k]
        next_item_ids = np.setdiff1d(next_item_ids, batch).tolist()
        return next_item_ids

    def configure_optimizers(self):
        if self.lr is None:
            self.lr = 1e-4
        if self.optimizer is None:
            if self.weight_decay is None:
                self.optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
            else:
                self.optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        if self.scheduler is None:
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, patience=10, factor=0.1)
        return {
            "monitor": "val_loss",
            "optimizer": self.optimizer,
            "lr_scheduler": self.scheduler
        }
