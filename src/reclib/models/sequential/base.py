from typing import Any, Dict, Optional, Tuple, Union

import pytorch_lightning as pl
import torch


class SequentialRecommender(pl.LightningModule):
    """
    A base class for sequential recommender systems.

    :param seq_length: The length of the input sequence.
    :param lr: The learning rate for the optimizer.
    :param weight_decay: The weight decay for the optimizer.
    :param optimizer: The optimizer to use.
    :param scheduler: The learning rate scheduler to use.
    :param args: Additional arguments (optional).
    :param kwargs: Additional keyword arguments (optional).
    """

    def __init__(
        self,
        seq_length: int = 120,
        lr: Optional[float] = None,
        weight_decay: Optional[float] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[torch.optim.lr_scheduler.LRScheduler] = None,
        *args: Any,
        **kwargs: Any,
    ):
        super().__init__()
        self.lr = lr
        self.seq_length = seq_length
        self.weight_decay = weight_decay
        self.optimizer = optimizer
        self.scheduler = scheduler

    def forward(self, item_ids: torch.Tensor, *args: Any) -> torch.Tensor:
        """
        The forward pass of the model.

        :param item_ids: The input sequence of item IDs.
        :param args: Additional arguments (optional).
        :raises NotImplementedError: not implemented method.
        """
        raise NotImplementedError()

    def handle_batch(
        self,
        batch: Any,
        inference: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """
        Handle a batch of data.

        :param batch: The batch of data.
        :param inference: Whether to run in inference mode or not.
        :raises NotImplementedError: not implemented method.
        """
        raise NotImplementedError()

    def batch_truths(self, batch: Any) -> torch.Tensor:
        """
        Get the ground truth labels from the batch.

        :param batch: The batch of data.
        :raises NotImplementedError: not implemented method.
        """
        raise NotImplementedError()

    def preprocess_infer_input(self, item_ids: torch.Tensor) -> torch.Tensor:
        """
        Preprocess the input for inference.

        :param item_ids: The input sequence of item IDs.
        :raises NotImplementedError: not implemented method.
        """
        raise NotImplementedError()

    def training_step(self, batch: Any, *args: Any) -> torch.Tensor:
        """
        The training step.

        :param batch: The batch of data.
        :param args: Additional arguments (optional).
        :return: The loss tensor.
        """
        _, loss, accuracy = self.handle_batch(batch)
        self.log("train_loss", loss, logger=True, on_step=True, on_epoch=True)
        self.log("train_accuracy", accuracy, logger=True, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch: Any, *args: Any) -> torch.Tensor:
        """
        The validation step.

        :param batch: The batch of data.
        :param args: Additional arguments (optional).
        :return: The loss tensor.
        """
        _, loss, accuracy = self.handle_batch(batch)
        self.log("val_loss", loss, logger=True, on_step=True, on_epoch=True)
        self.log("val_accuracy", accuracy, logger=True, on_step=True, on_epoch=True)
        return loss

    def test_step(
        self,
        batch: Any,
        *args: Any,
    ) -> Dict[str, Union[torch.Tensor, float]]:
        """
        The test step.

        :param batch: The batch of data.
        :param args: Additional arguments (optional).
        :return: A dictionary of metrics.
        """
        results: Dict[str, Union[torch.Tensor, float]] = {}
        predictions, loss, accuracy = self.handle_batch(batch)
        # predictions = torch.argsort(predictions[:, -1], dim=1, descending=True)
        # predictions = predictions.detach().cpu().numpy()
        # targets = batch[1][batch[0] == 1].detach().cpu().numpy()
        self.log("test_loss", loss, logger=True, on_step=True, on_epoch=True)
        self.log("test_accuracy", accuracy, logger=True, on_step=True, on_epoch=True)
        for metric, score in results.items():
            self.log(
                "test_{0}".format(metric),
                score,
                logger=True,
                on_step=True,
                on_epoch=True,
            )

        return results

    def predict_step(self, batch: Any, *args: Any) -> torch.Tensor:
        """
        The prediction step.

        :param batch: The batch of data.
        :param args: Additional arguments (optional).
        :return: The predictions.
        """
        predictions = self.handle_batch(batch, inference=True)
        assert isinstance(predictions, torch.Tensor)
        return torch.argsort(predictions[:, -1], dim=1, descending=True)

    def configure_optimizers(self) -> Dict[str, Any]:  # type: ignore
        """
        Configure the optimizers and learning rate schedulers.

        :return: A dictionary containing the optimizer and learning rate scheduler.
        """
        if self.lr is None:
            self.lr = 1e-4
        if self.optimizer is None:
            if self.weight_decay is None:
                self.optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
            else:
                self.optimizer = torch.optim.Adam(
                    self.parameters(),
                    lr=self.lr,
                    weight_decay=self.weight_decay,
                )
        if self.scheduler is None:
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                patience=10,
                factor=0.1,
            )
        return {
            "monitor": "val_loss",
            "optimizer": self.optimizer,
            "lr_scheduler": self.scheduler,
        }
