import hydra
import torch
import torchvision.transforms as T
from omegaconf import DictConfig


@hydra.main(version_base=None, config_path="configs", config_name="transform")
def get_default_transforms(conf: DictConfig):
    transform = {
        "train": T.Compose(
            [
                T.RandomHorizontalFlip(),
                T.RandomVerticalFlip(),
                T.RandomAffine(**conf.affine),
                T.ColorJitter(**conf.color_jitter),
                T.ConvertImageDtype(torch.float),
                T.Normalize(mean=conf.imagenet_mean, std=conf.imagenet_std),
            ]
        ),
        "val": T.Compose(
            [
                T.ConvertImageDtype(torch.float),
                T.Normalize(mean=conf.imagenet_mean, std=conf.imagenet_std),
            ]
        ),
    }
    return transform
