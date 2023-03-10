import random
import torchvision.transforms.functional as TF
from torchvision.transforms import ElasticTransform, InterpolationMode
import numpy as np

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

    def __init__(self, width, height, margin=0, uniform=True):
        self.width = width
        self.height = height
        self.margin = margin
        self.uniform = uniform

    def __call__(self, *arrays):
        img_height, img_width = arrays[0].squeeze().shape
        dx = img_width - self.width - 2 * self.margin
        dy = img_height - self.height - 2 * self.margin

        if self.uniform:
            top = self.margin + random.randint(0, dy - 1)
            left = self.margin + random.randint(0, dx - 1)
        else:
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

class CustomElasticTransform:
    """ 
    Perform the same elastic transform on multiple
    arrays (i.e. image and label)
    """
    def __init__(self, alpha, sigma):
        self.alpha = [alpha, alpha]
        self.sigma = [sigma, sigma]

    def __call__(self, *arrays):
        img_height, img_width = arrays[0].squeeze().shape
        displacement = ElasticTransform.get_params(self.alpha, self.sigma, [img_height, img_width])
        return [TF.elastic_transform(array, displacement, InterpolationMode.BILINEAR, 0) for array in arrays]