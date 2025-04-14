import os
import time
import torch
import numpy as np
from tqdm import tqdm
from torchutils.loggers.logger import logging
from torchutils.helpers import CheckpointHelper, MonitorHelper

class Trainer:
    
    def __init__(self, hparams, model, workspace_dir, multigpu=False, update_step=10):
        self._hparams = hparams
        self._handler_hparams = self._hparams.Handlers
        if not hasattr(self._handler_hparams, 'CheckpointHelper'):
            logging.error(f'[ trainer ] Missing "CheckpointHelper" in hparams!')
            raise KeyError
        if not hasattr(self._handler_hparams, 'MonitorHelper'):
            logging.error(f'[ trainer ] Missing "MonitorHelper" in hparams!')
            raise KeyError
        self._workspace_dir = workspace_dir
        self._model = model
        self._checkpoint_hparams = self._handler_hparams.CheckpointHelper
        self._monitor_hparams = self._handler_hparams.MonitorHelper
        self._save_every_step = self._handler_hparams.save_every_step
        self._scalar_save_every_step = self._handler_hparams.scalar_save_every_step
        self._valid_every_step = self._handler_hparams.valid_every_step
        self._max_steps = self._handler_hparams.max_steps
        self._max_epochs = self._handler_hparams.max_epochs
        self._model.setup_optimizer()
        self._multigpu = multigpu
        self._update_step = update_step
        if self._multigpu:
            self._model_module = self._model.module
        else:
            self._model_module = self._model
        
        self._checkpoint_helper = CheckpointHelper(
            self._checkpoint_hparams, 
            os.path.join(self._workspace_dir, 'checkpoints'), 
            self._model_module)
        self._train_monitor_helper = MonitorHelper(
            self._monitor_hparams, 
            os.path.join(self._workspace_dir, 'monitors', 'train'), 
            self._model_module)
        self._valid_monitor_helper = MonitorHelper(
            self._monitor_hparams, 
            os.path.join(self._workspace_dir, 'monitors', 'valid'), 
            self._model_module)

        self._train_dataloader = None
        self._valid_dataloader = None

        self._tqdm_bar = None

    def setup_dataloader(self, train_dataloader, valid_dataloader):
        self._train_dataloader = train_dataloader
        self._valid_dataloader = valid_dataloader

    def isMasterNode(self):
        return self._model_module.rank == 0 or str(self._model_module.rank) == 'cpu'

    def __call__(self):
        
        if self._train_dataloader is None or self._valid_dataloader is None:
            logging.error(f'[ trainer ] Trainer will not start until call setup_dataloader!')
            raise ValueError
        
        self._checkpoint_helper.load()
        if self._model_module.global_step >= self._max_steps or self._model_module.global_epoch >= self._max_epochs:
            logging.info(f'[ trainer ] Training is finished at Epoch {self._model_module.global_epoch}, Step {self._model_module.global_step}!')
            return
        
        finished = False

        if self.isMasterNode():
            self._tqdm_bar = tqdm(np.arange(self._model_module.global_step, self._max_steps + 1))
            
        while True:

            if self._multigpu:
                self._train_dataloader.sampler.set_epoch(self._model_module.global_epoch)
                self._valid_dataloader.sampler.set_epoch(self._model_module.global_epoch)
                
            for train_data in self._train_dataloader:
                st = time.time()
                train_outputs_dict, train_losses_dict, train_summaries_dict = self._forward_pass(train_data, is_training=True)
                et = time.time()
                
                if self.isMasterNode():
                    
                    if self._tqdm_bar is not None and self._model_module.global_step % self._update_step == 0:
                        self._print_losses(train_losses_dict, et - st, is_training=True)
                        self._tqdm_bar.update(self._update_step)
                    
                    if self._model_module.global_step > 0 and \
                            self._model_module.global_step % self._save_every_step == 0:
                        self._checkpoint_helper.save()
                        self._train_monitor_helper.save(train_summaries_dict)
                        
                    if self._model_module.global_step > 0 and \
                            self._model_module.global_step % self._scalar_save_every_step == 0:
                        self._train_monitor_helper.save({'scalar': {f'losses/{k}':v for k,v in train_losses_dict.items()}})
                        
                    with torch.no_grad():
                        if self._model_module.global_step > 0 and self._model_module.global_step % self._valid_every_step == 0:
                            valid_accum_losses_dict = {}
                            st = time.time()
                            for valid_data in self._valid_dataloader:
                                valid_outputs_dict, valid_losses_dict, valid_summaries_dict = self._forward_pass(valid_data, is_training=False)
                                for k, v in valid_losses_dict.items():
                                    if k not in valid_accum_losses_dict.keys():
                                        valid_accum_losses_dict[k] = []
                                    valid_accum_losses_dict[k].append(v)
                                break
                            for k, v in valid_accum_losses_dict.items():
                                valid_accum_losses_dict[k] = torch.stack(v).mean()
                            et = time.time()
                            self._valid_monitor_helper.save(valid_summaries_dict)
                            self._valid_monitor_helper.save({'scalar': valid_losses_dict})
                            self._print_losses(valid_accum_losses_dict, et - st, is_training=False)

                if self._model_module.global_step >= self._max_steps:
                    finished = True
                    break

            if finished:
                break

            self._model_module.epoch_done()

            if self._model_module.global_epoch >= self._max_epochs:
                break
        
        if self.isMasterNode():
            self._checkpoint_helper.save()

        logging.info(f'[ trainer ] Training is finished at Epoch {self._model_module.global_epoch}, Step {self._model_module.global_step}!')
            
    def _print_losses(self, losses_dict, time_cost, is_training):
        if self._tqdm_bar is None:
            logging.error(f'[ trainer ] Something wrong happened when initializing tqdm bar!')
            raise ValueError
        mode = 'Train' if is_training else 'Valid'
        info = f'[ {mode} ][ Epoch {self._model_module.global_epoch} - Step {self._model_module.global_step}] '
        for name, value in losses_dict.items():
            try:
                info += f'{name}={value:.3f}, '
            except:
                info += f'{name}={value}, '
        self._tqdm_bar.set_description_str(info)
        sys_info = self._train_monitor_helper.get_sys_info()
        sys_info['TC'] = time_cost
        self._tqdm_bar.set_postfix(sys_info)
        return info
        
    def _forward_pass(self, data, is_training):
        if is_training:
            self._model.train()
            self._model.set_mode(is_training, validating=False)
        else:
            self._model.eval()
            self._model.set_mode(is_training, validating=True)
        outputs_dict, losses_dict, summaries_dict = self._model(data)
        if is_training:
            self._model_module.update(losses_dict)
        return outputs_dict, losses_dict, summaries_dict
    