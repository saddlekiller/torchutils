import os
import torch
from torchutils.loggers.logger import logging
from .helpers import Helper

class CheckpointHelper(Helper):

    def __init__(self, hparams, save_dir, model):
        super().__init__(hparams=hparams, save_dir=save_dir, model=model)
        self._model_name = self._hparams.model_name
        self._max_to_keep = self._hparams.max_to_keep
        self._ckpts = []
        self._ckpt_fn = os.path.join(self._save_dir, 'checkpoint')

    def check_hparams(self):
        self._check_hparams("model_name")
        self._check_hparams("max_to_keep")
    
    def load(self, global_epoch=None, global_step=None):
        if not hasattr(self._model, 'load'):
            logging.error(f'[ helper-checkpoint ] function "save" NOT found in model "{__class__(self._model)}"')
            raise ValueError
        if global_epoch is None and global_step is None:
            ckpt = self._load_lastest_ckpt()
            if ckpt is None:
                logging.warn('[ helper-checkpoint ] Found NO checkpoints!')
            else:            
                logging.info(f'[ helper-checkpoint ] Loaded checkpoint from {ckpt}!')
        else:
            if global_epoch is None or global_step is None:
                logging.error('[ helper-checkpoint ] Must none or both global_epoch and global_step!')
                raise ValueError
            ckpt = os.path.join(self._save_dir, f'E{global_epoch}-S{global_step}_{self._model_name}.pth')
            if os.path.exists(ckpt):
                logging.info(f'[ helper-checkpoint ] Loaded checkpoint from {ckpt}!')
            else:
                logging.error(f'[ helper-checkpoint ] checkpoint "{ckpt}" NOT found!')
                raise FileNotFoundError
        if ckpt is not None:
            self._model.load(ckpt)
                
    def _load_lastest_ckpt(self):
        if os.path.exists(self._ckpt_fn):
            lines = [i.strip() for i in open(self._ckpt_fn).readlines()]
            ckpt = os.path.join(self._save_dir, lines[0].split('"')[1])
            self._ckpts = lines[1:]
            return ckpt
        else:
            return None
    
    def save(self):
        if not hasattr(self._model, 'global_epoch'):
            logging.error(f'[ helper-checkpoint ] "global_epoch" NOT found in model "{__class__(self._model)}"')
            raise ValueError
        if not hasattr(self._model, 'global_step'):
            logging.error(f'[ helper-checkpoint ] "global_step" NOT found in model "{__class__(self._model)}"')
            raise ValueError
        if not hasattr(self._model, 'save'):
            logging.error(f'[ helper-checkpoint ] function "save" NOT found in model "{__class__(self._model)}"')
            raise ValueError
        global_epoch = self._model.global_epoch
        global_step = self._model.global_step
        model_name = f'E{global_epoch}-S{global_step}_{self._model_name}.pth'
        self._ckpts.append(f'all_model_checkpoint_paths: "{model_name}"')
        ckpt = os.path.join(self._save_dir, model_name)
        if len(self._ckpts) > self._max_to_keep:
            for i in self._ckpts[:len(self._ckpts) - self._max_to_keep]:
                name = i.split('"')[1]
                os.remove(os.path.join(self._save_dir, name))
            self._ckpts = self._ckpts[len(self._ckpts) - self._max_to_keep:]
        with open(self._ckpt_fn, 'w') as f:
            f.write(f'model_checkpoint_path: "{model_name}"\n')
            for i in self._ckpts:
                f.write(f'{i}\n')
        self._model.save(ckpt)
        logging.info(f'[ helper-checkpoint ] Saved model at Epoch "{global_epoch}" Step "{global_step}" into "{ckpt}"')
    
