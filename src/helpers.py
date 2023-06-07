import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger


def get_trainer(epochs, device, log_dir, checkpoint_dir, model_prefix):
    logger = TensorBoardLogger(
        save_dir=log_dir,
    )
    checkpoint_handler = ModelCheckpoint(
        monitor="val_loss",
        mode="min",
        dirpath=checkpoint_dir,
        filename=f"{model_prefix}_best",
    )
    return pl.Trainer(
        accelerator=device,
        max_epochs=epochs,
        logger=logger,
        callbacks=[checkpoint_handler],
    )
