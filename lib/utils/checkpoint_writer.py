import re
import warnings
from collections import namedtuple
from os import listdir
from typing import Union

import torch
from pathlib import Path
from torch.nn import DataParallel

Checkpoint = namedtuple('Checkpoint', ['epoch', 'architecture', 'state_dict', 'optimizer', 'scheduler'])


class CheckpointsWriter:
    def __init__(self, model: torch.nn.Module, optimizer: torch.optim.Optimizer,
                 parent_folder: Union[str, Path], model_name: str,
                 scheduler: torch.optim.lr_scheduler._LRScheduler = None, interactive=True):
        self._model = model
        self._optimizer = optimizer
        self._scheduler = scheduler
        self._parent_folder = Path(parent_folder).joinpath(model_name)  # read-only, protected by property
        self._model_name = model_name  # read-only, protected by property
        self.interactive = interactive

        if self._parent_folder.exists():
            rewrite_accepted = self._assert_checkpoints_not_exist()
        else:
            rewrite_accepted = False
            self._parent_folder.mkdir(parents=True)
            print(f'Created checkpoints folder {self._parent_folder}')

        self._check_existing_checkpoints = not rewrite_accepted

    @property
    def full_path_template(self):
        return self._parent_folder.joinpath(self._model_name + '_{}.pth').as_posix()

    def save(self, epoch):
        if self._check_existing_checkpoints:
            self._assert_checkpoints_not_exist()
            self._check_existing_checkpoints = False

        full_path = self.full_path_template.format(epoch)

        if type(self._model) == DataParallel:
            model_state_dict = self._model.module.state_dict()
        else:
            model_state_dict = self._model.state_dict()
        optimizer_state_dict = self._optimizer.state_dict()

        saved_state = Checkpoint(epoch,
                                 architecture=self._model.module if type(self._model) == DataParallel else self._model,
                                 state_dict=model_state_dict,
                                 optimizer=optimizer_state_dict,
                                 scheduler=self._scheduler.state_dict() if self._scheduler else None)
        torch.save(saved_state, full_path)

    def load_checkpoint(self, epoch):
        full_path = self.full_path_template.format(epoch)
        save_state = torch.load(full_path)

        if type(self._model) == torch.nn.DataParallel:
            self._model.module.load_state_dict(save_state.state_dict)
        else:
            self._model.load_state_dict(save_state.state_dict)
        self._optimizer.load_state_dict(save_state.optimizer)

        if self._scheduler and save_state.scheduler:
            self._scheduler.load_state_dict(save_state.scheduler)
        elif self._scheduler:  # if we have a scheduler but not have a checkpoint to it
            warnings.warn("Checkpoint for scheduler does not exist!")

        if epoch != save_state.epoch:
            warnings.warn("The current and loaded epochs do not match!")

    def _assert_checkpoints_not_exist(self):
        r_exp = re.compile(rf'^{self._model_name}_\d+.pth')

        matched = [
            self._parent_folder.joinpath(item).as_posix()
            for item in listdir(self._parent_folder)
            if r_exp.match(item)
        ]

        rewrite_accepted = False
        if matched:
            if len(matched) > 10:
                m_pretty_repr = '\n'.join(matched[:10]) + '\n...'
            else:
                m_pretty_repr = '\n'.join(matched)

            msg = f'It seems that checkpoints for model {self._model_name} ' \
                  f'already exist in folder {self._parent_folder}:\n' \
                  f'{m_pretty_repr}'
            if self.interactive:
                response = input(msg + '\nProceed? y/n  ')
                if response.lower() not in ['y', 'ye', 'yes']:
                    print('Cancelled by user.')
                    exit(0)
                    return
                rewrite_accepted = True
            else:
                raise RuntimeError(msg)
        return rewrite_accepted

    @property
    def model_name(self):
        return self._model_name

    @property
    def parent_folder(self):
        return self._parent_folder
