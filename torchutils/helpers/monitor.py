import os
import psutil
import torch
import torchvision
import numpy as np
from tensorboardX import SummaryWriter
from torchutils.loggers.logger import logging
from .helpers import Helper

class MonitorHelper(Helper):

    def __init__(self, hparams, save_dir, model):
        super().__init__(hparams=hparams, save_dir=save_dir, model=model)
        self._sample_rate = self._hparams.sample_rate
        self._fps = self._hparams.fps
        self._event_save_dir = os.path.join(save_dir, 'events')
        self._vis_save_dir = os.path.join(save_dir, 'vis')
        os.makedirs(self._event_save_dir, exist_ok=True)
        os.makedirs(self._vis_save_dir, exist_ok=True)
        self._writer = SummaryWriter(self._event_save_dir)

    def get_sys_info(self):
        info = {
            'CPU MEM': psutil.Process().memory_info().rss / (1024**3),
            'CPU PER': psutil.cpu_percent()
        }
        if torch.cuda.is_available():
            gpu_mem = (torch.cuda.memory_allocated() + torch.cuda.memory_reserved()) / (1024**3)
            info['GPU MEM'] = gpu_mem
        return info

    def check_hparams(self):
        self._check_hparams('sample_rate')
        self._check_hparams('fps')
    
    def save(self, tensor_dict=None, mode='events'):
        match (mode):
            case "events":
                self._save_to_events(self._model.global_epoch, self._model.global_step, tensor_dict)
                sys_info = {f'sysinfo/{k}': v for k, v in self.get_sys_info().items()}
                self._save_to_events(self._model.global_epoch, self._model.global_step, {'scalar': sys_info})
            case "files":
                self._save_to_files(self._model.global_epoch, self._model.global_step, tensor_dict)
            case _:
                logging.error(f'[ helper-monitor ] Invalid mode "{mode}", should be in ["events", "files"]')
                raise ValueError

    def _save_to_events(self, global_epoch=None, global_step=None, tensor_dict=None):
        for tensor_type, tensor_info in tensor_dict.items():
            for tensor_name, tensor_with_func in tensor_info.items():
                tensor = self._process_tensor_with_func(tensor_name, tensor_with_func)
                match (tensor_type):
                    case "scalar":
                        tensor = self._process_scalar(tensor_name, tensor)
                        self._writer.add_scalar(tensor_name, tensor, global_step)
                    case "image":
                        tensor = self._process_image(tensor_name, tensor)
                        self._writer.add_image(tensor_name, tensor, global_step)
                    case "histogram":
                        tensor = self._process_histogram(tensor_name, tensor)
                        self._writer.add_histogram(tensor_name, tensor, global_step)
                    case "audio":
                        tensor = self._process_audio(tensor_name, tensor)
                        self._writer.add_audio(tensor_name, tensor, global_step, sample_rate=self._sample_rate)
                    case "video":
                        tensor = self._process_video(tensor_name, tensor)
                        self._writer.add_video(tensor_name, tensor, global_step, fps=self._fps)                
                    case _:
                        logging.error(f'[ helper-monitor ] Invalid type for "{tensor_name}": {tensor_type}')
                        raise NotImplementedError
        self._writer.flush()

    def _save_to_files(self, global_epoch=None, global_step=None, tensor_dict=None):
        save_subdir = os.path.join(self._vis_save_dir, f'E{global_epoch}-S{global_step}')
        os.makedirs(save_subdir, exist_ok=True)
        for tensor_type, tensor_info in tensor_dict.items():
            for tensor_name, tensor_with_func in tensor_info.items():
                tensor = self._process_tensor_with_func(tensor_name, tensor_with_func)
                match (tensor_type):
                    case "image":
                        if tensor.shape[1] not in [1, 3]:
                            logging.error(f'[ helper-monitor ] Invalid channel number "{tensor.shape[1]}", should be 1 or 3')
                        torchvision.utils.save_image(tensor, os.path.join(save_subdir, tensor_name + '.jpg'))
                    case _:
                        pass
    
    def _process_scalar(self, tensor_name, tensor):
        if type(tensor) not in [float, int]:
            shape = tensor.shape
            if len(shape) != 0:
                logging.error(f'[ helper-monitor ] Invalid shape "{shape}" for "{tensor_name}" which is "scalar"!')
                raise ValueError
        return tensor
    
    def _process_image(self, tensor_name, tensor):
        shape = tensor.shape
        if len(shape) == 4:
            # B x C x H x W
            tensor = torchvision.utils.make_grid(tensor, int(np.ceil(tensor.shape[0] ** 0.5)), normalize=True, value_range=(0, 1))
        elif len(shape) ==  3:
            # B x C x T
            tensor = tensor[0, None]
        elif len(shape) == 2:
            tensor = tensor[None]
        else:
            logging.error(f'[ helper-monitor ] Invalid shape "{shape}" for "{tensor_name}" which is "image"!')
            raise ValueError
        return tensor[:3]

    def _process_histogram(self, tensor_name, tensor):
        shape = tensor.shape
        # if len(shape) < 2:
        #     logging.error(f'[ helper-monitor ] Invalid shape "{shape}" for "{tensor_name}" which is "histogram"!')
        #     raise ValueError
        return tensor
    
    def _process_audio(self, tensor_name, tensor):
        shape = tensor.shape
        if len(shape) == 2:
            tensor = tensor[0, None]
        elif len(shape) == 1:
            tensor = tensor[None]
        else:
            logging.error(f'[ helper-monitor ] Invalid shape "{shape}" for "{tensor_name}" which is "audio"!')
            raise ValueError
        return tensor

    def _process_video(self, tensor_name, tensor):
        shape = tensor.shape
        if len(shape) == 5:
            tensor = tensor[0, :, :3]
        elif len(shape) == 4:
            tensor = tensor[None, :, :3]
        else:
            logging.error(f'[ helper-monitor ] Invalid shape "{shape}" for "{tensor_name}" which is "video"!')
            raise ValueError
        return tensor

    def _process_tensor_with_func(self, tensor_name, tensor_with_func):
        if type(tensor_with_func) not in [list, tuple]:
            return tensor_with_func
        else:
            if len(tensor_with_func) == 1:
                tensor = tensor_with_func[0]
                tensor_func = lambda x: x
            elif len(tensor_with_func) == 2:
                tensor, tensor_func = tensor_with_func
            else:
                logging.error(f'[ helper-monitor ] Invalid format for "{tensor_name}", should be either list or tuple has length 1 or 2, the first element represents tensor and the other is processing function!')
                raise ValueError
            return tensor_func(tensor)

    def __del__(self):
        self._writer.close()
