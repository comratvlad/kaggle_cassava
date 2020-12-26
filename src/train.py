"""
sudo PYTHONPATH='/home/researcher/cassava' HYDRA_FULL_ERROR=1 /home/researcher/miniconda3/bin/python3.7 train.py -cp ../data/configs -cn example +device=0
"""
import hydra
import torch
from omegaconf import DictConfig

from lib.utils.config_parser import ConfigParser


@hydra.main()
def main(cfg: DictConfig) -> None:
    settings = ConfigParser(cfg)
    train_loader = settings.train_loader
    model = settings.model
    model.to(settings.device)
    loss = settings.loss
    optimizer = settings.optimizer
    for batch in train_loader:
        model_input = batch[settings.model_input_feature].type(torch.FloatTensor).to(settings.device)
        model_output = model(model_input)
        weighted_sum, components_eval = loss(batch, model_output)
        optimizer.zero_grad()
        weighted_sum.backward()
        optimizer.step()
        print(components_eval)


if __name__ == '__main__':
    main()
