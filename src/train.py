"""
sudo PYTHONPATH='/home/researcher/cassava' HYDRA_FULL_ERROR=1 /home/researcher/miniconda3/bin/python3.7 train.py -cp ../data/configs -cn example +device=0
"""
import hydra
import torch
from omegaconf import DictConfig
from collections import defaultdict

from lib.utils.config_parser import ConfigParser


def train_one_epoch(model, model_input_feature, loader, loss, optimizer, device):
    model.train()
    component_values = defaultdict(int)
    for batch in loader:
        model_input = batch[model_input_feature].type(torch.FloatTensor).to(device)
        model_output = model(model_input)
        weighted_sum, _components = loss(batch, model_output)
        component_values = {k: val + component_values[k] for k, val in _components.items()}
        optimizer.zero_grad()
        weighted_sum.backward()
        optimizer.step()
    component_values = {k: i / len(loader) for k, i in component_values.items()}
    return component_values


@hydra.main()
def main(cfg: DictConfig) -> None:
    settings = ConfigParser(cfg)
    train_loader = settings.train_loader
    model = settings.model
    model.to(settings.device)
    loss = settings.loss
    optimizer = settings.optimizer

    # Training loop
    for epoch in range(settings.n_epochs):
        # Train ========================================================================================================
        train_losses = train_one_epoch(model, settings.model_input_feature,
                                       train_loader, loss, optimizer, settings.device)

        print('Epoch: {}'.format(epoch), end='; ')
        for loss_name in train_losses:
            print('{}: {:.9f}; '.format(loss_name, train_losses[loss_name]))
        # ==============================================================================================================


if __name__ == '__main__':
    main()
