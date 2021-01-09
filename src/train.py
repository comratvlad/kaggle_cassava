from collections import defaultdict
from pathlib import Path

import clearml
import hydra
import torch
from omegaconf import DictConfig
from tqdm import tqdm

from lib.losses.weighted_sum_loss import WeightedSumLoss
from lib.utils.checkpoint_writer import CheckpointsWriter
from lib.utils.config_parser import ConfigParser


def train_one_epoch(model, model_input_feature, loader, loss, optimizer, device):
    model.train()
    component_values = defaultdict(int)
    for batch in tqdm(loader):
        model_input = batch[model_input_feature].type(torch.FloatTensor).to(device)
        model_output = model(model_input)
        weighted_sum, _components = loss(batch, model_output)
        component_values = {k: val + component_values[k] for k, val in _components.items()}
        optimizer.zero_grad()
        weighted_sum.backward()
        optimizer.step()
    component_values = {k: i / len(loader) for k, i in component_values.items()}
    return component_values


def evaluate(loss: WeightedSumLoss, model, loaders, model_input_feature, metrics, torch_device):
    """
    :param loss: an instance of WeightedSumLoss for evaluating
    :param model: pyTorch model to evaluate
    :param loaders: a dictionary in format {<dataset name>: <pyTorch DataLoader instance>, ...}
    :param model_input_feature: a string that will be used as a key to get model input from sampled data-batch
    :param metrics: a dictionary {<metric name>: <metric function import>}. Metric function is a callable that takes
    predictions as the first argument and true labels as the second
    :param torch_device: device to place model input to (must be the same device as the model's)
    :return: a dictionary { <metric name>: { <dataset name>: <metric result>, ...}, ...}
    """

    metrics_result = {metric_name: {dataset_name: 0 for dataset_name in loaders.keys()} for metric_name in metrics}
    loss_dict = {dataset_name: {} for dataset_name in loaders.keys()}
    model.eval()

    for dataset_name, loader in loaders.items():
        pred_proba = []
        ground_truth = []
        with torch.no_grad():
            for batch in tqdm(loader):
                model_input = batch[model_input_feature].type(torch.FloatTensor).to(torch_device)
                model_output = model(model_input)
                weighted_sum, _components = loss(batch, model_output)
                if not loss_dict[dataset_name]:
                    loss_dict[dataset_name].update(_components)
                else:
                    loss_dict[dataset_name] = {k: val + _components[k] for k, val in loss_dict[dataset_name].items()}
                ground_truth.extend(list(batch['disease_label'].detach().cpu()))
                pred_proba.extend(list(model_output['disease_prediction'].detach().cpu()))
        loss_dict[dataset_name] = {k: i / len(loader) for k, i in loss_dict[dataset_name].items()}
        for metric_name, metric_function in metrics.items():
            metrics_result[metric_name][dataset_name] = metric_function(ground_truth, pred_proba)
    return metrics_result, loss_dict


@hydra.main()
def main(cfg: DictConfig) -> None:
    settings = ConfigParser(DictConfig(cfg))
    train_loader = settings.train_loader
    dev_loaders = settings.dev_loaders
    model = settings.model
    model.to(settings.device)
    loss = settings.loss
    optimizer = settings.optimizer
    metrics_dict = settings.metrics_dict

    task = clearml.Task.init(settings.experiment, settings.task, auto_resource_monitoring=False)
    logger = task.get_logger()

    if settings.checkpoints:
        _p_folder = Path(settings.checkpoints).joinpath(settings.experiment)
        checkpoints_writer = CheckpointsWriter(model, optimizer, _p_folder, settings.task,
                                               settings.scheduler)
    else:
        checkpoints_writer = None

    # Training loop
    for epoch in range(settings.n_epochs):
        # Train ========================================================================================================
        train_losses = train_one_epoch(model, settings.model_input_feature,
                                       train_loader, loss, optimizer, settings.device)
        print('Epoch: {}'.format(epoch), end='; ')
        for loss_name in train_losses:
            print('{}: {:.6f}; '.format(loss_name, train_losses[loss_name]))
            logger.report_scalar(title='{} train'.format(loss_name), series='Loss',
                                 value=train_losses[loss_name], iteration=epoch)
        # ==============================================================================================================

        # Dev ==========================================================================================================
        eval_results, loss_dict = evaluate(loss, model, dev_loaders, settings.model_input_feature,
                                           metrics_dict, settings.device)
        for dataset_name in loss_dict:
            print(dataset_name+':')
            for loss_name in loss_dict[dataset_name]:
                print('{}: {:.9f}; '.format(loss_name, loss_dict[dataset_name][loss_name]))
                logger.report_scalar(title='{} {}'.format(loss_name, dataset_name), series='Loss',
                                     value=loss_dict[dataset_name][loss_name], iteration=epoch)

        for metric_name, metrics_result in eval_results.items():
            print('{}: {}'.format(metric_name, ' '.join(['{} - {:.4f}'.format(dataset_name, value)
                                                         for dataset_name, value in metrics_result.items()])))
            for dataset_name, value in metrics_result.items():
                logger.report_scalar(title='{} {}'.format(metric_name, dataset_name), series='Metric',
                                     value=value, iteration=epoch)
        # ==============================================================================================================

        if settings.scheduler:
            print('lr: {:.6f}'.format(optimizer.param_groups[0]['lr']))
            logger.report_scalar(title='Learning rate', series='Learning rate',
                                 value=optimizer.param_groups[0]['lr'], iteration=epoch)
            settings.scheduler.step()

        if checkpoints_writer:
            checkpoints_writer.save(epoch)


if __name__ == '__main__':
    main()
