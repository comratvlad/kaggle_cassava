import pydoc
from functools import partial
from typing import Union, Dict

import albumentations as A
import hydra
from torch.nn import Module
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.data import DataLoader, ConcatDataset

from lib.losses.weighted_sum_loss import WeightedSumLoss, LossSettings
from lib.utils.data.dataset import FolderDataset


class ConfigParser:
    loss: WeightedSumLoss
    dev_loaders: Dict[str, DataLoader]
    train_loader: DataLoader

    optimizer: Optimizer
    scheduler: Union[_LRScheduler, None]
    checkpoints: str
    model: Module
    model_input_feature: str

    def __init__(self, config):
        self.experiment = config.experiment
        self.task = config.task
        self.train_loader = self._get_train_loader(config.train_data, config.sampled_features, config.transforms,
                                                   config.augmentations, config.batch_size, config.num_workers)
        self.dev_loaders = self._get_dev_loaders(config.dev_data, config.sampled_features, config.dev_transforms,
                                                 config.batch_size, config.num_workers)
        self.model = hydra.utils.instantiate(config.model)
        self.model_input_feature = config.model_input_feature
        self.optimizer = hydra.utils.instantiate(config.optimizer, params=self.model.parameters())
        self.scheduler = hydra.utils.instantiate(config.scheduler, optimizer=self.optimizer) if 'scheduler' in config \
            else None
        self.loss = self._get_weighted_sum_loss(config.losses, config.device)
        self.device = config.device
        self.n_epochs = config.n_epochs
        self.metrics_dict = {name: pydoc.locate(value) for name, value in config.metrics.items()}
        self.checkpoints = config.checkpoints if 'checkpoints' in config else None

    @staticmethod
    def get_normalize():
        transform = A.Compose([
            A.CenterCrop(480, 480),
            A.Normalize()
        ])
        return transform

    @staticmethod
    def get_weak_augmentations():
        transform = A.Compose([
            A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.50, rotate_limit=45, p=.75),
            A.RandomCrop(480, 480),
            A.Normalize()
        ])
        return transform

    @staticmethod
    def get_hard_augmentations():
        transform = A.Compose([
            A.CLAHE(),
            A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.50, rotate_limit=45, p=.75),
            A.Blur(blur_limit=3),
            A.OpticalDistortion(),
            A.GridDistortion(),
            A.HueSaturationValue(),
            A.RandomCrop(480, 480),
            A.Normalize()
        ])
        return transform

    @staticmethod
    def _get_train_loader(train_data, sampled_features, transforms, augmentations, batch_size, num_workers):
        train_datasets = []
        for _, train_dataset_setting in train_data.items():
            transforms = pydoc.locate(transforms)
            # TODO: full configure augmentations by config.yaml
            if augmentations == 'weak':
                transforms_with_augmentations = partial(transforms,
                                                        albumentations_compose=ConfigParser.get_weak_augmentations())
            else:
                transforms_with_augmentations = partial(transforms,
                                                        albumentations_compose=ConfigParser.get_hard_augmentations())
            dataset = FolderDataset(train_dataset_setting.path, train_dataset_setting.info_path,
                                    features=[pydoc.locate(feature) for feature in sampled_features],
                                    transforms=transforms_with_augmentations, filter_by=train_dataset_setting.filter_by)
            train_datasets.append(dataset)

        train_dataset = ConcatDataset(train_datasets)
        train_loader = DataLoader(train_dataset,
                                  batch_size=batch_size,
                                  shuffle=True,
                                  num_workers=num_workers)
        return train_loader

    @staticmethod
    def _get_dev_loaders(dev_data, sampled_features, transforms, batch_size, num_workers):
        dev_loaders = {}
        for name, dev_dataset_setting in dev_data.items():
            transforms = pydoc.locate(transforms)
            # TODO: full configure augmentations by config.yaml
            transforms_with_augmentations = partial(transforms, albumentations_compose=ConfigParser.get_normalize())
            dataset = FolderDataset(dev_dataset_setting.path, dev_dataset_setting.info_path,
                                    features=[pydoc.locate(feature) for feature in sampled_features],
                                    transforms=transforms_with_augmentations, filter_by=dev_dataset_setting.filter_by)
            dev_loaders[name] = DataLoader(dataset,
                                           batch_size=batch_size,
                                           num_workers=num_workers)
        return dev_loaders

    @staticmethod
    def _get_weighted_sum_loss(losses_description, device):
        components = []
        try:
            for name, description in losses_description.items():
                # TODO: this shit doesn't work with containers as an arguments :(
                loss_instance = hydra.utils.instantiate(description.callable)
                components.append(LossSettings(
                    name=name,
                    instance=loss_instance,
                    args=description.args,
                    weight=description.weight
                ))
        except KeyError as e:
            raise ValueError(f'Missing key "{e.args[0]}" in losses description') from e
        return WeightedSumLoss(*components, device=device)
