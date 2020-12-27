import pydoc

import hydra
from torch.nn import Module
from torch.optim import Optimizer
from torch.utils.data import DataLoader, ConcatDataset

from lib.losses.weighted_sum_loss import WeightedSumLoss, LossSettings
from lib.utils.data.dataset import FolderDataset


class ConfigParser:
    loss: WeightedSumLoss
    # dev_loaders: Dict[str, DataLoader]
    train_loader: DataLoader

    optimizer: Optimizer
    # scheduler: Union[_LRScheduler, None]
    # checkpoints: str
    # tensorboard: TensorboardSettings
    model: Module
    model_input_feature: str

    def __init__(self, config):
        self.train_loader = self._get_train_loader(config.train_data, config.sampled_features,
                                                   config.batch_size, config.num_workers)
        self.model = hydra.utils.instantiate(config.model)
        self.model_input_feature = config.model_input_feature
        self.optimizer = hydra.utils.instantiate(config.optimizer, params=self.model.parameters())
        self.loss = self._get_weighted_sum_loss(config.losses, config.device)
        self.device = config.device
        self.n_epochs = config.n_epochs

    @staticmethod
    def _get_train_loader(train_data, sampled_features, batch_size, num_workers):
        train_datasets = []
        for _, train_dataset_setting in train_data.items():
            dataset = FolderDataset(train_dataset_setting.path, train_dataset_setting.info_path,
                                    features=[pydoc.locate(feature) for feature in sampled_features],
                                    transforms=None)
            train_datasets.append(dataset)

        train_dataset = ConcatDataset(train_datasets)
        train_loader = DataLoader(train_dataset,
                                  batch_size=batch_size,
                                  shuffle=True,
                                  # sampler=train_sampler,
                                  num_workers=num_workers)
        return train_loader

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
