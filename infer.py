import os
import torch
import pandas as pd
from petfinder.model import Model
from petfinder.config import config
from petfinder.dataset import PetfinderDataModule

if __name__ == '__main__':
    df = pd.read_csv(os.path.join(config.root, "test.csv"))
    ids = df['Id'].copy()
    df["Id"] = df["Id"].apply(lambda x: os.path.join(config.root, "test", x + ".jpg"))

    model = Model(config) 
    model.load_state_dict(torch.load(f'{config.model.name}/default/version_0/checkpoints/best_loss.ckpt')['state_dict'])
    model = model.cuda().eval()
    config.val_loader.batch_size = 16
    datamodule = PetfinderDataModule(None, df, config)
    preds = model.predict(datamodule.val_dataloader())
    preds = pd.DataFrame({'Id' : ids, 'y' : preds})
    preds.to_csv('solution.csv')