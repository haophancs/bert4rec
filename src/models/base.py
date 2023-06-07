import torch
import pytorch_lightning as pl


class SequentialRecommender(pl.LightningModule):
    def __init__(self, lr=None, weight_decay=None, optimizer=None, scheduler=None, *args, **kwargs):
        super(SequentialRecommender, self).__init__()
        if lr is None and optimizer is None:
            raise ValueError()
        if lr is not None and optimizer is not None:
            raise ValueError()
        self.lr = lr
        self.weight_decay = weight_decay
        self.optimizer = optimizer
        self.scheduler = scheduler

    def forward(self, item_ids, *args):
        raise NotImplementedError()

    def handle_batch(self, batch):
        raise NotImplementedError()

    def training_step(self, batch, *args):
        loss, accuracy = self.handle_batch(batch)
        self.log("train_loss", loss)
        self.log("train_accuracy", accuracy)
        return loss

    def validation_step(self, batch, *args):
        loss, accuracy = self.handle_batch(batch)
        self.log("val_loss", loss)
        self.log("val_accuracy", accuracy)
        return loss

    def test_step(self, batch, *args):
        loss, accuracy = self.handle_batch(batch)
        self.log("test_loss", loss)
        self.log("test_accuracy", accuracy)
        return loss

    def configure_optimizers(self):
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
