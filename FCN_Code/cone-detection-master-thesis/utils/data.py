"""
Utility script to provide functions for creating datasets
from a given set of cone maps + annotations
"""

import os
import glob
from tqdm import tqdm
import numpy as np
from scipy.io import loadmat
from skimage.io import imread, imsave

import torch

from utils.image import class_to_dt, erode_image
from utils.transforms import RotationTransform, CropTransform

def create_split(input_path, 
    output_path, 
    cross_validation=True, 
    fraction_validation=0.2, 
    fraction_test=0.2,
    erode_images=False,
    black_list=[]):
    """
    Create a Train-Validation-Test split of a given set 
    of images and labels
    """

    if not os.path.exists(output_path):
        # Get all available subjects
        subjects = sorted(os.listdir(input_path))
        n = len(subjects)
        indices = np.random.permutation(n)
        num_test = int(np.round(fraction_test * n))

        # Initiate parameters depending on cross_validation
        if cross_validation:
            assert fraction_test < 1.0
            assert fraction_test > 0.0

            types = ["test", "train"]
            indices_per_type = [indices[:num_test], indices[num_test:]]
        else:
            assert fraction_validation + fraction_test < 1.0
            assert fraction_validation + fraction_test > 0.0

            num_validation = int(np.round(fraction_validation * n))
            types = ["validation", "test", "train"]
            indices_per_type = [indices[:num_validation], indices[num_validation:num_validation+num_test], indices[num_validation+num_test:]]

        # Enumerate subsets
        for idx, type in enumerate(types):    
            # Create subfolder  
            os.makedirs(os.path.join(output_path,type)) 

            # Create .npz files
            for index in indices_per_type[idx]:
                # Check for blacklist and eventually continue
                if subjects[index] in black_list:
                    continue

                # Load .mat file
                files = sorted(os.listdir(os.path.join(input_path, subjects[index])))  
                mat_file = files[1] # First file = image, second file = mat
                mat_file_path = os.path.join(input_path, subjects[index], mat_file)
                mat = loadmat(mat_file_path)

                # Get relevant properties from .mat file and transform (x,y) MATLAB-1-based-coordinates
                # to (y,x) Python-0-based-coordinates in a 2D-grid
                image = mat["I"]
                locations = np.round(mat["conelocs"][:,:2] - 1).astype(int)
                label = np.zeros(image.shape)
                label[locations[:,1], locations[:,0]] = 1 # labels are in (x,y) format instead of (y,x)!
                cdc20 = mat["euclidianNCones"]["CDC20_loc"][0,0] # Get array from object
                background_probability = np.sum(label == 0)
                cone_probability = np.sum(label == 1)
                background_probability, cone_probability = background_probability / (background_probability + cone_probability), cone_probability / (background_probability + cone_probability)

                # Optionally erode image (and apply to label)
                if erode_images:
                    image, applied_to = erode_image(image, apply_to=[label])
                    label = applied_to[0]

                # Save .npz file
                npz_file_path = os.path.join(output_path, type, f"{subjects[index]}.npz")
                np.savez(npz_file_path, image=image, label=label, cdc20=cdc20, background_probability=background_probability, cone_probability=cone_probability)
    else:
        print(f"Split {output_path} has already been created")

def generate_dataset_offline(input_path, 
    output_path, 
    cross_validation=True,
    crop_size=(256,256), 
    iterations=100, 
    margin=0,
    convert_to_dt=False,
    for_reference=False,
    uniform_crop=True,
    erode_images=False):
    """
    Use this class to iterate a directory of (images, ground_truth)
    which shall be processed offline to be used in a training /
    validation / test dataset
    """

    if not os.path.exists(output_path):
        # Set available image types
        types = ["test", "train"] if cross_validation else ["validation", "test", "train"]

        # Get crop transform
        crop_transform = CropTransform(crop_size[0], crop_size[1], margin=margin, uniform=uniform_crop)

        # Iterate types
        for type in types:
            # Create subfolder  
            type_path = os.path.join(output_path, type)
            os.makedirs(type_path)

            # Create folders for training a reference network (Hamwood et al., 2019)
            # with this dataset
            if for_reference:
                type_images_path = os.path.join(type_path, "imgs")
                if not os.path.exists(type_images_path):
                    os.mkdir(type_images_path)

                type_labels_path = os.path.join(type_path, "truth")
                if not os.path.exists(type_labels_path):
                    os.mkdir(type_labels_path)

            # Get subjects per type
            input_type_path = os.path.join(input_path, type, "*")
            subjects = sorted(glob.glob(input_type_path)) 

            # Iterate subjects
            for subject_id, subject in tqdm(enumerate(subjects)):
                # Load npz file
                npz_file_path = os.path.join(input_path, type, subject)
                npz = np.load(npz_file_path)

                # Get properties from file
                image = npz["image"]
                label = npz["label"]
                # cdc20 = npz["cdc20"]

                # Optionally erode image (and apply to label)
                if erode_images:
                    image, applied_to = erode_image(image, apply_to=[label])
                    label = applied_to[0]

                # Image stats
                background_probability = np.sum(label == 0)
                cone_probability = np.sum(label == 1)
                background_probability, cone_probability = background_probability / (background_probability + cone_probability), cone_probability / (background_probability + cone_probability)

                # Optionally convert label to DT representation
                if convert_to_dt:
                    label = class_to_dt(label) 

                # Optionally remove distance information from background of the stitched AOSLO image
                # if dt_remove_black:
                #     label[image == 0] = 0

                # Create tensors
                image = torch.Tensor(image)
                label = torch.Tensor(label) # FUCK. This was previously torch.Tensor(image) :'(

                # Create n cropped images
                for iteration in range(iterations): 
                    image_crop, label_crop = crop_transform(image, label)
                    type_path_crop = os.path.join(type_path, f"{(subject_id * iterations + iteration):06}.npz")
                    image_crop, label_crop = image_crop.numpy().astype(np.uint8), label_crop.numpy().astype(np.uint8)
                    np.savez(type_path_crop, image=image_crop, label=label_crop, background_probability=background_probability, cone_probability=cone_probability)

                    if for_reference:
                        # Store files also in Hamwood et al. format
                        imsave(os.path.join(type_images_path, f"{(subject_id * iterations + iteration)}.png"), image_crop, check_contrast=False)
                        imsave(os.path.join(type_labels_path, f"{(subject_id * iterations + iteration)}.png"), label_crop + 1, check_contrast=False)


    else:
        print(f"Dataset {output_path} has already been generated")