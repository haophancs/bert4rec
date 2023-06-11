import numpy as np
import pytorch_lightning as pl
import torch

from src.recsys.metrics.sequential import hr_at_k


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

    def batch_truths(self, batch):
        raise NotImplementedError()

    def training_step(self, batch, *args):
        _, loss, accuracy = self.handle_batch(batch)
        self.log("train_loss", loss, logger=True, on_step=True, on_epoch=True)
        self.log("train_accuracy", accuracy, logger=True, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch, *args):
        _, loss, accuracy = self.handle_batch(batch)
        self.log("val_loss", loss, logger=True, on_epoch=True)
        self.log("val_accuracy", accuracy, logger=True, on_epoch=True)
        return loss

    def test_step(self, batch, *args):
        truths = self.batch_truths(batch)
        predictions, loss, accuracy = self.handle_batch(batch, inference=False)
        self.log("test_loss", loss, logger=True, on_epoch=True)
        self.log("test_accuracy", accuracy, logger=True, on_epoch=True)
        for metric, score in hr_at_k(predictions, truths, [1, 5, 10]).items():
            self.log(f"test_{metric}", score / len(batch), logger=True, on_epoch=True)
        return loss

    def predict_step(self, batch, *args, k=10):
        with torch.no_grad():
            predictions = self.handle_batch(batch, inference=True)
        next_item_ids = predictions[0, -1].detach().cpu().numpy()
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
