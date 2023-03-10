import torch
import numpy as np
from skimage.io import imread
import matplotlib.pyplot as plt
from scipy import ndimage

def get_mask(image : np.ndarray, threshold = 1/255.0):
    """
    Get a mask of an image which excludes the black
    background
    """
    # Compute mask by flooding from all 4 sides
    top = np.cumsum(image, 0) < threshold
    bottom = np.flip(np.cumsum(np.flip(image, [0]), 0), [0]) < threshold
    left = np.cumsum(image, 1) < threshold
    right = np.flip(np.cumsum(np.flip(image, [1]), 1), [1]) < threshold
    # Only include pixels where all directions are False (= not part of the black border)
    mask = ~(top | bottom | left | right) 
    return mask

def mask_array(array : np.ndarray, image : np.ndarray, better_mask, threshold = 1/255.0, fill=0):
    """
    Mask an array either naively (= remove all black pixels) or
    with more computational effort (= flood fill from borders)
    """
    if better_mask:
        # Flooded from 4 sides
        mask = get_mask(image)
    else:
        # Just exclude all black pixels
        mask = (image >= threshold).float() 

    return array * mask + (1 - mask) * fill    

def image_to_tensor(image : np.ndarray):
    """
    Convert a (H,W) array to a (B,C,H,W) tensor
    """
    return torch.Tensor(image).unsqueeze(0).unsqueeze(0)

def tensor_to_image(tensor : torch.Tensor):
    """
    Convert a (B,C,H,W) tensor to a (H,W) array
    """
    return tensor.squeeze(0).squeeze(0).numpy()

def clip_image(image : np.ndarray, clipping_factor=8):
    """
    Clip an image to a shape divisible by clipping_factor
    """
    clip_y, clip_x = image.shape[0] % clipping_factor, image.shape[1] % clipping_factor
    if clip_y > 0 and clip_x > 0:
        image = image[:-clip_y,:-clip_x]
    elif clip_y > 0:
        image = image[:-clip_y,:]
    elif clip_x > 0:
        image = image[:,:-clip_x]

    return image

def erode_image(image : np.ndarray, iterations=64, better_mask=True, threshold=1/255.0, apply_to=[]):
    """
    Perform erosion on the given image to counteract imaging
    artifacts at the borders of the image
    """
    if better_mask:
        mask = get_mask(image, threshold=threshold)
        mask = ndimage.binary_erosion(mask, iterations=iterations)
    else:
        # TODO: Not sure if this will create holes in images 
        # which have black pixels
        mask = ndimage.binary_erosion(image, iterations=iterations)
    return image * mask, [other * mask for other in apply_to]

def read_image(image_path : str):
    """
    Read a gray-scale retina image
    """
    return imread(image_path, as_gray=True)

def mark_cones(image : np.ndarray, mask : np.ndarray):
    """
    Mark cone locations in an image
    """
    color_image = np.stack([image, image, image])
    color_image[0,mask == 1] = 0
    color_image[1,mask == 1] = 255
    color_image[2,mask == 1] = 0
    return color_image.transpose(1,2,0)

def show_image(image : np.ndarray):
    """
    Show an image using matplotlib.pyplot
    """
    _ = plt.imshow(image) #, cmap="gray")
    plt.show()

def class_to_dt(image : np.ndarray):
    """
    Convert binary class (background <> cone) to 
    a distance transform representation
    """
    return ndimage.distance_transform_edt(1.0 - image, return_distances=True)

def dt_to_class(image : np.ndarray):
    """
    Convert a distance transform to a binary class
    (background <> cone) representation
    """
    return np.uint8(image == 0)