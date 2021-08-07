import pydoc
from functools import partial
from typing import Union, Dict

import albumentations as A
import hydra
import omegaconf
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
                                                 config.dev_augmentations, config.batch_size, config.num_workers)
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
    def _make_albumentations_pipeline(description):
        pipeline = []
        for item in description:
            if isinstance(item, dict) or omegaconf.OmegaConf.is_config(item):
                if len(item) != 1:
                    raise ValueError(f'String or dictionary with single key containing import string is expected '
                                     f'in every list item; got {item}')
                import_str, params = list(item.items())[0]
            else:
                import_str = item
                params = {}
            pipeline.append(pydoc.locate(import_str)(**params))
        return A.Compose(pipeline)

    @staticmethod
    def _get_train_loader(train_data, sampled_features, transforms, augmentations, batch_size, num_workers):
        train_datasets = []
        for _, train_dataset_setting in train_data.items():
            transforms = pydoc.locate(transforms)
            if augmentations:
                transforms = partial(transforms,
                                     albumentations_compose=ConfigParser._make_albumentations_pipeline(augmentations))
            dataset = FolderDataset(train_dataset_setting.path, train_dataset_setting.info_path,
                                    features=[pydoc.locate(feature) for feature in sampled_features],
                                    transforms=transforms, filter_by=train_dataset_setting.filter_by)
            train_datasets.append(dataset)

        train_dataset = ConcatDataset(train_datasets)
        train_loader = DataLoader(train_dataset,
                                  batch_size=batch_size,
                                  shuffle=True,
                                  num_workers=num_workers)
        return train_loader

    @staticmethod
    def _get_dev_loaders(dev_data, sampled_features, transforms, augmentations, batch_size, num_workers):
        dev_loaders = {}
        for name, dev_dataset_setting in dev_data.items():
            transforms = pydoc.locate(transforms)
            if augmentations:
                transforms = partial(transforms,
                                     albumentations_compose=ConfigParser._make_albumentations_pipeline(augmentations))
            dataset = FolderDataset(dev_dataset_setting.path, dev_dataset_setting.info_path,
                                    features=[pydoc.locate(feature) for feature in sampled_features],
                                    transforms=transforms, filter_by=dev_dataset_setting.filter_by)
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
