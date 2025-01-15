import os
from loggers.logger import logging

class Helper:
    
    def __init__(self, hparams, save_dir, model=None):
        self._hparams = hparams
        self._save_dir = save_dir
        self._model = model
        self.check_hparams()
        os.makedirs(self._save_dir, exist_ok=True)
        
    def _check_hparams(self, name):
        if not hasattr(self._hparams, name):
            logging.error(f'[ helper ] Missing "{name}" in {self._hparams}')
            raise KeyError
        
    def check_hparams(self):
        raise NotImplementedError
    
    def __call__(self):
        raise NotImplementedError
    
    