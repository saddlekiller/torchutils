import os
import time
import torch
import pickle
import numpy as np
from tqdm import tqdm
from torchutils.loggers.logger import logging
from torchutils.helpers import CheckpointHelper, MonitorHelper

class Inferer:
    
    def __init__(self, hparams, model, workspace_dir):
        self._hparams = hparams
        if not hasattr(hparams, 'CheckpointHelper'):
            logging.error(f'[ trainer ] Missing "CheckpointHelper" in hparams!')
            raise KeyError
        self._workspace_dir = workspace_dir
        self._model = model
        self._checkpoint_hparams = self._hparams.CheckpointHelper
        self._monitor_hparams = self._hparams.MonitorHelper
        self._outputs_save_dir = os.path.join(self._workspace_dir, 'infer', 'outputs')
        self._summaries_save_dir = os.path.join(self._workspace_dir, 'infer', 'summaries')

        self._checkpoint_helper = CheckpointHelper(
            self._checkpoint_hparams, 
            os.path.join(self._workspace_dir, 'checkpoints'), 
            self._model)
        self._monitor_helper = MonitorHelper(
            self._monitor_hparams, 
            self._summaries_save_dir, 
            self._model)
        
        self._checkpoint_helper.load()
        self._model.eval()
        self._model.set_mode(training=False, validating=False)
        
        os.makedirs(self._outputs_save_dir, exist_ok=True)
        os.makedirs(self._summaries_save_dir, exist_ok=True)
        
    def __call__(self, data, mode):
        if 'name' not in data.keys():
            logging.error(f'[ inferer ] Invalid data format, Missing "name"!')
            raise KeyError
        with torch.no_grad():
            name = data['name']
            outputs_dict, _, summaries_dict = self._model(data, mode)
            summary_images = summaries_dict['image']
            renamed_summary_images = {}
            for k, v in summary_images.items(): 
                renamed_summary_images[f'{k}-{name}'] = v
            self._monitor_helper.save({'image': renamed_summary_images}, mode='files', subdir_name=mode)
            for k, v in outputs_dict.items():
                pickle.dump(v.cpu().detach().numpy(), open(os.path.join(self._outputs_save_dir, k + '.pkl'), 'wb'))    
            return outputs_dict

        
        