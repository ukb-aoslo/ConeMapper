import numpy as np
import torch
import os

def get_circular_mask(height, width, center_y, center_x, distance_min, distance_max, tensor=False):
    """
    Get a circular mask centered at (center_y, center_x)
    to mask out all position which are further away than 
    distance_max and closer than distance_min. Optionally return a tensor
    """
    y = np.linspace(-center_y, height - center_y - 1, height)
    x = np.linspace(-center_x, width - center_x - 1, width)
    xv, yv = np.meshgrid(x, y)

    dist = xv ** 2 + yv ** 2
    d_min = distance_min ** 2
    d_max = distance_max ** 2

    dist[dist < d_min] = 0.0
    dist[dist > d_max] = 0.0
    dist[dist != 0.0] = 1.0

    if tensor:
        return torch.Tensor(dist)
    else:
        return dist

def arcmin_to_pixels(pixels_per_degree, arcminutes):
    """
    Convert arcminutes to pixels using the 
    given resolution of pixels per degree
    of visual field
    """
    return arcminutes / 60.0 * pixels_per_degree

def pixels_to_arcmin(pixels_per_degree, pixels):
    """
    Convert pixels to arcminutes using the 
    given resolution of pixels per degree
    of visual field
    """
    return pixels / pixels_per_degree * 60.0

def get_maximum_extent(height, width, center_y, center_x, pixels_per_degree):
    """
    Get the maximum extent possible given
    the specified center
    """
    d = np.min([center_x, width - center_x, center_y, height - center_y])
    return pixels_to_arcmin(pixels_per_degree, d)

def read_cdc(path, filename="cdc20.csv"):
    """
    Read CDC csv file generated using MATLAB script
    """
    cdc20_path = os.path.join(path, filename)
    return np.genfromtxt(cdc20_path, delimiter=",", skip_header=1)