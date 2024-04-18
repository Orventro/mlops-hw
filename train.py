import os

import hydra
import pandas as pd
import pytorch_lightning as pl
from omegaconf import DictConfig
from pytorch_lightning import callbacks
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from sklearn.model_selection import StratifiedKFold

from petfinder.dataset import PetfinderDataModule
from petfinder.model import Model


@hydra.main(version_base=None, config_path="petfinder/configs", config_name="global")
def train(config: DictConfig):
    df = pd.read_csv(os.path.join(config.root, config.train))
    df["Id"] = df["Id"].apply(
        lambda x: os.path.join(config.root, config.train_dir, x + ".jpg")
    )

    skf = StratifiedKFold(
        n_splits=config.n_splits, shuffle=True, random_state=config.seed
    )

    for train_idx, val_idx in skf.split(df["Id"], df["Pawpularity"]):
        train_df = df.loc[train_idx].reset_index(drop=True)
        val_df = df.loc[val_idx].reset_index(drop=True)
        datamodule = PetfinderDataModule(train_df, val_df, config)
        model = Model(config)
        earystopping = EarlyStopping(monitor="val_loss")
        lr_monitor = callbacks.LearningRateMonitor()
        loss_checkpoint = callbacks.ModelCheckpoint(
            filename="best_loss",
            monitor="val_loss",
            save_top_k=1,
            mode="min",
            save_last=False,
        )
        logger = TensorBoardLogger(config.model.name)

        trainer = pl.Trainer(
            logger=logger,
            max_epochs=config.epoch,
            callbacks=[lr_monitor, loss_checkpoint, earystopping],
            **config.trainer,
        )
        trainer.fit(model, datamodule=datamodule)


if __name__ == "__main__":
    train()
