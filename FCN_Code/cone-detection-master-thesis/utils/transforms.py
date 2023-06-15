import random
import torch
import torchvision.transforms.functional as TF
from torchvision.transforms import ElasticTransform, InterpolationMode
import numpy as np
from scipy import ndimage

from utils.coordinates import polar_to_cartesian
from utils.distribution import uniform_2D_sphere
from utils.image import erode_image, image_to_tensor, tensor_to_image

class RotationTransform:
    """ 
    Rotate an image by one of the given angles
    """

    def __init__(self, angles):
        self.angles = angles

    def __call__(self, *arrays):
        angle = random.choice(self.angles)
        return [TF.rotate(array, angle) for array in arrays]

class CropTransform:
    """ 
    Get a random crop of size (height, width) of an image
    """

    def __init__(self, width, height, margin=0, mode="uniform_rectangle"):
        self.width = width
        self.height = height
        self.margin = margin
        self.mode = mode

    def __call__(self, *arrays):
        img_height, img_width = arrays[0].squeeze().shape
        dx = img_width - self.width - 2 * self.margin
        dy = img_height - self.height - 2 * self.margin

        if self.mode.lower() == "uniform_rectangle":
            top = self.margin + random.randint(0, dy - 1)
            left = self.margin + random.randint(0, dx - 1)
        elif self.mode.lower() == "uniform_sphere":
            polar_pos = uniform_2D_sphere(1, min(dx // 2, dy // 2))
            origin = np.array([img_height // 2, img_width // 2])
            top, left = polar_to_cartesian(polar_pos, origin=origin)[0]
            top, left = int(top - self.width // 2), int(left - self.width // 2)  
            #print(img_height, img_width, top, left)      
        elif self.mode.lower() == "normal":
            mean_x = dx // 2
            mean_y = dy // 2
            std_x = dx // 6
            std_y = dy // 6
            left = int(self.margin + mean_x + std_x * np.random.randn())
            top = int(self.margin + mean_y + std_y * np.random.randn())

        return [TF.crop(array, top, left, self.height, self.width) for array in arrays]

class ClipTransform:
    """
    Clip a tensor (B,C,H,W) to a (H,W) shape divisible by clipping_factor
    """
    def __init__(self, clipping_factor=8):
        self.clipping_factor = clipping_factor

    def __call__(self, image):
        img_height, img_width = image.squeeze().squeeze().shape
        clip_y, clip_x = img_height % self.clipping_factor, img_width % self.clipping_factor
        if clip_y > 0 and clip_x > 0:
            image = image[:,:,:-clip_y,:-clip_x]
        elif clip_y > 0:
            image = image[:,:,:-clip_y,:]
        elif clip_x > 0:
            image = image[:,:,:,:-clip_x]

        return image

class ErodeTransform:
    """
    Perform erosion on the given image to counteract imaging
    artifacts at the borders of the image
    """
    def __init__(self, iterations=64, threshold=1/255.0):
        self.iterations = iterations
        self.threshold = threshold

    def __call__(self, *arrays):
        arrays = [tensor_to_image(a) for a in arrays]
        image, applied = erode_image(arrays[0], self.iterations, threshold=self.threshold, apply_to=arrays[1:])
        image = image_to_tensor(image)
        return image, [image_to_tensor(a) for a in applied]

class LabelRegularityTransform:
    """
    Transform a label in a regularity preserving way

    TPR: Percentage [0,1] of kept cones
    FDR: Percentage [0,1] of randomly generated cones
    """
    def __init__(self, tpr=0.98, fdr=0.005, cone_probability=0.02):
        self.tpr = tpr
        self.fdr = fdr
        self.cone_p = cone_probability

    def __call__(self, array):
        background_p = 1.0 - self.cone_p

        # Purge some true cone locations
        tpr_filter = torch.rand_like(array) < self.tpr * background_p
        # Add some false cone locations
        fdr_filter = torch.rand_like(array) < self.fdr * self.cone_p

        result = array * tpr_filter + fdr_filter

        return result
    
class DistanceTransform:
    """
    Convert binary class representation to distance transform
    """
    def __init__(self):
        pass

    def __call__(self, array):
        batch_size = array.shape[0]
        for b in range(batch_size):
            array[b,0,:,:] = torch.from_numpy(ndimage.distance_transform_edt(1.0 - array[b,0,:,:], return_distances=True))
        return array


class RandomElasticTransform:
    """ 
    Perform the same random elastic transform on multiple
    arrays (i.e. image and label). 

    Alpha is sampled with a standard normal distribution
    (mean = 0, std = 1) whose absolute value is taken and divided by
    3 to guarantee by 99.6% that the chosen value is in range
    [0, max_alpha]. Default sigma is 1.0
    """
    def __init__(self, max_alpha = 4.0, sigma = 1.0):
        self.max_alpha = max_alpha
        self.sigma = sigma

    def __call__(self, *arrays):
        alpha_n = np.abs(np.random.randn()) / 3.0 # 99.6% of values in [0,1]
        alpha = [alpha_n * self.max_alpha, alpha_n * self.max_alpha]
        sigma = [self.sigma, self.sigma]
        img_height, img_width = arrays[0].squeeze().shape
        displacement = ElasticTransform.get_params(alpha, sigma, [img_height, img_width])
        return [TF.elastic_transform(array, displacement, InterpolationMode.BILINEAR, 0) for array in arrays]