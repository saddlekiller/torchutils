
import torch
import torch.nn as nn

class AbstractNetwork(nn.Module):

    def __init__(self, hparams, device):
        super().__init__()
        self.global_step = 0
        self.global_epoch = 0
        self._optimizer = None
        self._scheduler = None
        self._hparams = hparams
        self.validating = False
        if type(device) == int:
            if device < 0:
                # CPU mode
                self._device = torch.device('cpu')
            else:
                self._device = device
        else:
            self._device = device
        self._setup_parameters()
    
    @property
    def rank(self):
        return self._device
    
    def _setup_parameters(self):
        raise NotImplementedError
    
    def forward(self, data, mode=None):
        raise NotImplementedError
    
    def update(self, losses_dict):
        raise NotImplementedError
    
    def epoch_done(self):
        self.global_epoch += 1
        
    def load(self, ckpt):
        if ckpt is not None:
            state_dict = torch.load(ckpt)
            self.global_epoch = state_dict['global_epoch']
            self.global_step = state_dict['global_step']
            self.load_state_dict(state_dict['parameters'])

    def save(self, fn):
        state_dict = {
            'global_epoch': self.global_epoch,
            'global_step': self.global_step,
            'parameters': self.state_dict()
        }
        torch.save(state_dict, fn)
    
    def set_mode(self, training, validating):
        self.validating = validating

    def setup_optimizer(self):
        raise NotImplementedError