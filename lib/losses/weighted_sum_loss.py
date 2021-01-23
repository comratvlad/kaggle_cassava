from collections import namedtuple
from copy import copy

import torch

LossSettings = namedtuple('LossSettings', ['instance', 'name', 'weight', 'args'])


class WeightedSumLoss(torch.nn.Module):
    def __init__(self, *components: LossSettings, device):
        super(WeightedSumLoss, self).__init__()
        self._components = components
        self.device = device

    def forward(self, model_output, batch):
        # TODO: inspect or remember saved_args = locals()
        components_eval = {}
        weighted_sum = 0
        for component in self._components:
            loss_inputs = {}
            for name, params in component.args.items():
                # TODO: what if there are no key?
                # TODO: check the format
                source_name, *key = params.split('.')
                if len(key) == 0:
                    loss_inputs[name] = locals()[source_name].to(self.device)
                elif len(key) == 1:
                    loss_inputs[name] = locals()[source_name][key[0]].to(self.device)
                else:
                    raise ValueError('Wrong format of losses.args.')
            loss_value: torch.Tensor = component.instance(**loss_inputs)
            weighted_sum += component.weight * loss_value
            components_eval[component.name] = loss_value.item()
        return weighted_sum, components_eval

    @property
    def components(self):
        return copy(self._components)
