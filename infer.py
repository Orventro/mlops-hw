import os

import hydra
import pandas as pd
import torch
from omegaconf import DictConfig

from petfinder.dataset import PetfinderDataModule
from petfinder.model import Model


@hydra.main(version_base=None, config_path="petfinder/configs", config_name="global")
def infer(config: DictConfig):
    df = pd.read_csv(os.path.join(config.root, config.test))
    ids = df["Id"].copy()
    df["Id"] = df["Id"].apply(
        lambda x: os.path.join(config.root, config.test_dir, x + ".jpg")
    )

    model = Model(config)
    model.load_state_dict(
        torch.load(os.path.join(config.model.name, config.best_ckpt))["state_dict"]
    )
    model = model.cuda().eval()
    config.val_loader.batch_size = 16
    datamodule = PetfinderDataModule(None, df, config)
    preds = model.predict(datamodule.val_dataloader())
    preds = pd.DataFrame({"Id": ids, "y": preds})
    preds.to_csv(config.output)


if __name__ == "__main__":
    infer()
