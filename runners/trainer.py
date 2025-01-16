import os
import time
import torch
import numpy as np
from tqdm import tqdm
from loggers.logger import logging
from helpers import CheckpointHelper, MonitorHelper

class Trainer:
    
    def __init__(self, hparams, model, workspace_dir):
        self._hparams = hparams
        if not hasattr(hparams, 'CheckpointHelper'):
            logging.error(f'[ trainer ] Missing "CheckpointHelper" in hparams!')
            raise KeyError
        if not hasattr(hparams, 'MonitorHelper'):
            logging.error(f'[ trainer ] Missing "MonitorHelper" in hparams!')
            raise KeyError
        self._workspace_dir = workspace_dir
        self._model = model
        self._checkpoint_hparams = self._hparams.CheckpointHelper
        self._monitor_hparams = self._hparams.MonitorHelper
        self._save_every_step = self._hparams.save_every_step
        self._valid_every_step = self._hparams.valid_every_step
        self._max_steps = self._hparams.max_steps
        self._max_epochs = self._hparams.max_epochs

        self._checkpoint_helper = CheckpointHelper(
            self._checkpoint_hparams, 
            os.path.join(self._workspace_dir, 'checkpoints'), 
            self._model)
        self._train_monitor_helper = MonitorHelper(
            self._monitor_hparams, 
            os.path.join(self._workspace_dir, 'monitors', 'train'), 
            self._model)
        self._valid_monitor_helper = MonitorHelper(
            self._monitor_hparams, 
            os.path.join(self._workspace_dir, 'monitors', 'valid'), 
            self._model)

        self._train_dataloader = None
        self._valid_dataloader = None
        self._tqdm_bar = None

    def setup_dataloader(self, train_dataloader, valid_dataloader):
        self._train_dataloader = train_dataloader
        self._valid_dataloader = valid_dataloader

    def __call__(self):
        if self._train_dataloader is None or self._valid_dataloader is None:
            logging.error(f'[ trainer ] Trainer will not start until call setup_dataloader!')
            raise ValueError
        self._checkpoint_helper.load()
        if self._model.global_step >= self._max_steps or self._model.global_epoch >= self._max_epochs:
            logging.info(f'[ trainer ] Training is finished at Epoch {self._model.global_epoch}, Step {self._model.global_step}!')
            return
        finished = False
        self._tqdm_bar = tqdm(np.arange(self._model.global_step, self._max_steps + 1))
        while True:
            for train_data in self._train_dataloader:
                st = time.time()
                train_outputs_dict, train_losses_dict, train_summaries_dict = self._forward_pass(train_data, is_training=True)
                et = time.time()    
                self._print_losses(train_losses_dict, et - st, is_training=True)
                
                if self._model.global_step > 0 and self._model.global_step % self._save_every_step == 0:
                    self._checkpoint_helper.save()
                    self._train_monitor_helper.save(train_summaries_dict)
                    self._train_monitor_helper.save({'scalar': train_losses_dict})
                    
                if self._model.global_step > 0 and self._model.global_step % self._valid_every_step == 0:
                    valid_accum_losses_dict = {}
                    st = time.time()
                    for valid_data in self._valid_dataloader:
                        valid_outputs_dict, valid_losses_dict, valid_summaries_dict = self._forward_pass(valid_data, is_training=False)
                        for k, v in valid_losses_dict.items():
                            if k not in valid_accum_losses_dict.keys():
                                valid_accum_losses_dict[k] = []
                            valid_accum_losses_dict[k].append(v)
                    for k, v in valid_accum_losses_dict.items():
                        valid_accum_losses_dict[k] = torch.stack(v).mean()
                    et = time.time()
                    self._valid_monitor_helper.save(valid_summaries_dict)
                    self._valid_monitor_helper.save({'scalar': valid_losses_dict})
                    self._print_losses(valid_accum_losses_dict, et - st, is_training=False)

                if self._model.global_step >= self._max_steps:
                    finished = True
                    break
            if finished:
                break
            self._model.epoch_done()
            if self._model.global_epoch >= self._max_epochs:
                break

        self._checkpoint_helper.save()
        logging.info(f'[ trainer ] Training is finished at Epoch {self._model.global_epoch}, Step {self._model.global_step}!')
            
    def _print_losses(self, losses_dict, time_cost, is_training):
        mode = 'Train' if is_training else 'Valid'
        info = f'[ {mode} ][ Epoch {self._model.global_epoch} - Step {self._model.global_step}] '
        for name, value in losses_dict.items():
            info += f'{name}={value:.3f}, '
        self._tqdm_bar.set_description_str(info)
        # info += f'[ '
        # for name, value in self._train_monitor_helper.get_sys_info().items():
        #     info += f'{name}={value:.2f}, '
        # info += f'TC={time_cost:.2f}s ]'
        sys_info = self._train_monitor_helper.get_sys_info()
        sys_info['TC'] = time_cost
        self._tqdm_bar.set_postfix(sys_info)
        return info
        
    def _forward_pass(self, data, is_training):
        if is_training:
            self._model.train()
        else:
            self._model.eval()
        outputs_dict, losses_dict, summaries_dict = self._model(data)
        if is_training:
            self._model.update(losses_dict)
            self._tqdm_bar.update()
        return outputs_dict, losses_dict, summaries_dict
    