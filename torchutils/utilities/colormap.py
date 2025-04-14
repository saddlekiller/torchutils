import torch
from matplotlib import cm
from torchutils.loggers.logger import logging

def apply_colormap(image, cmap="viridis"):
    if image.shape[1] != 1:
        logging.error(f'[ colormap ] Invalid channel number "{image.shape[1]}", should be 1!')
        raise ValueError
    colormap = cm.get_cmap(cmap)
    colormap = torch.tensor(colormap.colors).to(image.device)  # type: ignore
    image_long = (image * 255).long()
    image_long_min = torch.min(image_long)
    image_long_max = torch.max(image_long)
    assert image_long_min >= 0, f"the min value is {image_long_min}"
    assert image_long_max <= 255, f"the max value is {image_long_max}"
    new_shape = list(image_long.shape)
    new_shape = [new_shape[0]] + new_shape[2:]
    return colormap[image_long[:, 0, :, :].reshape([-1])].reshape(new_shape + [3]).permute([0, 3, 1, 2])

def apply_depth_colormap(depth, cmap="turbo", min=None, max=None):
    near_plane = float(torch.min(depth)) if min is None else min
    far_plane = float(torch.max(depth)) if max is None else max
    depth = (depth - near_plane) / (far_plane - near_plane + 1e-10)
    depth = torch.clip(depth, 0, 1)
    colored_image = apply_colormap(depth, cmap=cmap)
    return colored_image

class TensorColorMap:
    
    @staticmethod
    def depth_to_rgb(tensor):
        return apply_depth_colormap(tensor, min=0., max=1.)
    
    @staticmethod
    def normal_to_rgb(tensor):
        return (tensor + 1) / 2
    
    @staticmethod
    def rescale(tensor):
        return (tensor - tensor.min()) / torch.clamp(tensor.max() - tensor.min(), min=1e-10)

    