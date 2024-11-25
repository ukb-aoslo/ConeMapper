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
import matplotlib.pyplot as plt
from skimage import exposure
from skimage.morphology import disk
from skimage.filters import rank
import scipy.ndimage as nd

import torch

from utils.cdc import get_bounding_box
from utils.coordinates import polar_to_cartesian
from utils.distribution import uniform_2D_sphere, uniform_2D_rectangle
from utils.image import class_to_dt, erode_image, clip_image
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
        for idx, data_type in enumerate(types):    
            # Create subfolder  
            os.makedirs(os.path.join(output_path,data_type)) 

            # Create .npz files
            for index in indices_per_type[idx]:
                # Check for blacklist and eventually continue
                if subjects[index] in black_list:
                    continue

                # Load .mat file
                files = sorted(os.listdir(os.path.join(input_path, subjects[index])))  
                mat_file = files[0] # First file = image, second file = mat
                mat_file_path = os.path.join(input_path, subjects[index], mat_file)
                mat = loadmat(mat_file_path)

                # Get relevant properties from .mat file and transform (x,y) MATLAB-1-based-coordinates
                # to (y,x) Python-0-based-coordinates in a 2D-grid
                image = mat["I"]

                # Save exact locations
                #exact_npz_file_path = os.path.join(output_path, data_type, f"{subjects[index]}_exact.npz")
                exact_gt = mat["conelocs"][:,:2] - 1
                exact_gt[:,[0,1]] = exact_gt[:,[1,0]] # (x,y) to (y,x)
                #np.savez(exact_npz_file_path, cones=exact_gt)

                # Prepare label for CNN
                locations = np.round(mat["conelocs"][:,:2] - 1).astype(int)
                label = np.zeros(image.shape)
                label[locations[:,1], locations[:,0]] = 1 # labels are in (x,y) format instead of (y,x)!
                try:
                    cdc20 = mat["euclideanNCones"]["CDC20_loc"][0,0] # Get array from object
                except:
                    cdc20 = [[0,0]]
                background_probability = np.sum(label == 0)
                cone_probability = np.sum(label == 1)
                background_probability, cone_probability = background_probability / (background_probability + cone_probability), cone_probability / (background_probability + cone_probability)

                # Optionally erode image (and apply to label)
                if erode_images:
                    image, applied_to = erode_image(image, apply_to=[label])
                    label = applied_to[0]

                # Save .npz file
                npz_file_path = os.path.join(output_path, data_type, f"{subjects[index]}.npz")
                np.savez(npz_file_path, image=image, label=label, cdc20=cdc20, exact_gt=exact_gt, background_probability=background_probability, cone_probability=cone_probability)
    else:
        print(f"Dataset {output_path} has already been created")

def noise_dataset(input_path,
    output_path,
    sigma):
    """
    Noise the dataset
    """
    if not os.path.exists(output_path):
        # Set available image types
        types = ["test", "train"]

        # Iterate types
        for type in types:
            # Create subfolder  
            output_type_path = os.path.join(output_path, type)
            os.makedirs(output_type_path)

            # Get subjects per type
            input_type_path = os.path.join(input_path, type)
            subjects = sorted(os.listdir(input_type_path))

            # Iterate subjects
            for subject_id, subject in tqdm(enumerate(subjects)):
                # Load npz file
                npz_file_path = os.path.join(input_type_path, subject)
                npz = np.load(npz_file_path)

                # Get properties from file
                image = npz["image"]
                label = npz["label"]

                # Gaussian noise
                mask = (image != 0) # Only apply noise to montage
                noise = 0.0 + sigma * np.random.randn(image.shape[0], image.shape[1])
                image = image + mask * noise

                # Save .npz file
                output_npz_file_path = os.path.join(output_type_path, subject)
                np.savez(output_npz_file_path, image=image, label=label, cdc20=npz["cdc20"], exact_gt=npz["exact_gt"],
                         background_probability=npz["background_probability"], cone_probability=npz["cone_probability"])
    else:
        print(f"Dataset {output_path} has already been created")

def blur_dataset(input_path,
    output_path,
    sigma):
    """
    Blur the dataset
    """
    if not os.path.exists(output_path):
        # Set available image types
        types = ["test", "train"]

        # Iterate types
        for type in types:
            # Create subfolder  
            output_type_path = os.path.join(output_path, type)
            os.makedirs(output_type_path)

            # Get subjects per type
            input_type_path = os.path.join(input_path, type)
            subjects = sorted(os.listdir(input_type_path))

            # Iterate subjects
            for subject_id, subject in tqdm(enumerate(subjects)):
                # Load npz file
                npz_file_path = os.path.join(input_type_path, subject)
                npz = np.load(npz_file_path)

                # Get properties from file
                image = npz["image"]
                label = npz["label"]

                # Blur image
                image = nd.gaussian_filter(image, sigma=sigma)

                # Save .npz file
                output_npz_file_path = os.path.join(output_type_path, subject)
                np.savez(output_npz_file_path, image=image, label=label, cdc20=npz["cdc20"], exact_gt=npz["exact_gt"],
                         background_probability=npz["background_probability"], cone_probability=npz["cone_probability"])
    else:
        print(f"Dataset {output_path} has already been created")

def equalize_dataset(input_path,
    output_path,
    radius=16):
    """
    Locally equalize the histogram of a dataset
    """
    if not os.path.exists(output_path):
        # Set available image types
        types = ["test", "train"]

        # Iterate types
        for type in types:
            # Create subfolder  
            output_type_path = os.path.join(output_path, type)
            os.makedirs(output_type_path)

            # Get subjects per type
            input_type_path = os.path.join(input_path, type)
            subjects = sorted(os.listdir(input_type_path))

            # Iterate subjects
            for subject_id, subject in tqdm(enumerate(subjects)):
                # Load npz file
                npz_file_path = os.path.join(input_type_path, subject)
                npz = np.load(npz_file_path)

                # Get properties from file
                image = npz["image"]
                label = npz["label"]

                # Local histogram equalization
                selem = disk(radius)
                image = rank.equalize(image, footprint=selem, mask=(image > 0))

                # Save .npz file
                output_npz_file_path = os.path.join(output_type_path, subject)
                np.savez(output_npz_file_path, image=image, label=label, cdc20=npz["cdc20"], exact_gt=npz["exact_gt"],
                         background_probability=npz["background_probability"], cone_probability=npz["cone_probability"])
    else:
        print(f"Dataset {output_path} has already been created")

def erode_dataset(input_path,
    output_path,
    iterations=64):
    """
    Erode images of a dataset
    """
    if not os.path.exists(output_path):
        # Set available image types
        types = ["test", "train"]

        # Iterate types
        for type in types:
            # Create subfolder  
            output_type_path = os.path.join(output_path, type)
            os.makedirs(output_type_path)

            # Get subjects per type
            input_type_path = os.path.join(input_path, type)
            subjects = sorted(os.listdir(input_type_path))

            # Iterate subjects
            for subject_id, subject in tqdm(enumerate(subjects)):
                # Load npz file
                npz_file_path = os.path.join(input_type_path, subject)
                npz = np.load(npz_file_path)

                # Get properties from file
                image = npz["image"]
                label = npz["label"]

                # Erode image (and apply to label)
                image, applied_to = erode_image(image, iterations=iterations, apply_to=[label])
                label = applied_to[0]

                # Save .npz file
                output_npz_file_path = os.path.join(output_type_path, subject)
                np.savez(output_npz_file_path, image=image, label=label, cdc20=npz["cdc20"], exact_gt=npz["exact_gt"],
                         background_probability=npz["background_probability"], cone_probability=npz["cone_probability"])
    else:
        print(f"Dataset {output_path} has already been created")

def crop_dataset(input_path,
    output_path,
    crop_size=256,
    iterations=256,
    visualize_sampling=False,
    types=["test", "train"],
    save_distance_cdc=True):
    """
    Crop images of a dataset
    """
    if not os.path.exists(output_path):
        # Get crop transform
        # mode = "uniform_sphere" if crop_mode == None else crop_mode
        # crop_transform = CropTransform(crop_size[0], crop_size[1], margin=margin, mode=mode)

        # Iterate types
        for type in types:
            # Create subfolder  
            output_type_path = os.path.join(output_path, type)
            os.makedirs(output_type_path)

            # Get subjects per type
            input_type_path = os.path.join(input_path, type)
            subjects = sorted(os.listdir(input_type_path))

            # Iterate subjects
            for subject_id, subject in tqdm(enumerate(subjects)):
                # Load npz file
                npz_file_path = os.path.join(input_type_path, subject)
                npz = np.load(npz_file_path)

                # Get properties from file
                image = npz["image"]
                label = npz["label"]
                if save_distance_cdc:
                    cdc = npz["cdc20"][0]
                    cy, cx = cdc[0], cdc[1]

                # height, width = image.shape
                # cy, cx = height // 2, width // 2 #cdc[0], cdc[1] #

                offset = crop_size // 2
                bbox_min, bbox_max = get_bounding_box(image)
                bbox_min, bbox_max = np.array(bbox_min), np.array(bbox_max)
                bbox_min = np.max([[offset, offset], bbox_min], axis=0)
                bbox_max = np.min([[image.shape[0] - offset, image.shape[1] - offset], bbox_max], axis=0)
                locations = uniform_2D_rectangle(iterations * 2, bbox_min, bbox_max) # Twice should be enough for rejection sampling 

                # horizontal_radius = max(cx - bbox_min[1], bbox_max[1] - cx)
                # vertical_radius = max(cy - bbox_min[0], bbox_max[0] - cy)
                # radius = max(horizontal_radius, vertical_radius)
                # offset = crop_size // 2
                # locations = uniform_2D_sphere(iterations * 2, radius - offset) # Twice should be enough for rejection sampling
                # locations = polar_to_cartesian(locations, cdc)

                if save_distance_cdc:
                    # Build distance map for different dilations
                    dmap_x, dmap_y = np.meshgrid(np.arange(0, image.shape[1], 1), np.arange(0, image.shape[0], 1))
                    dmap_x = dmap_x - cx
                    dmap_y = dmap_y - cy
                    distance_map = np.sqrt(dmap_y ** 2 + dmap_x ** 2)

                if visualize_sampling:
                    def visualize_particles(particles, image=None):
                        y, x = particles[:,0], particles[:,1]
                        plt.figure(dpi=200)
                        plt.scatter(x, y, c="blue", s=0.2)
                        if image is not None:
                            plt.imshow(image, cmap="gray")
                        plt.show(block=True)

                    visualize_particles(locations, image)

                # Create n cropped images
                index = 0
                for iteration in range(iterations): 
                    #crop_y, crop_x = -1, -1
                    #while crop_y < bbox_min[0] or crop_y > bbox_max[0] or crop_x < bbox_min[1] or crop_x > bbox_max[1]:
                    crop_y, crop_x = locations[index, :]
                    index += 1

                    image_crop = image[int(crop_y-offset):int(crop_y+offset),int(crop_x-offset):int(crop_x+offset)]
                    label_crop = label[int(crop_y-offset):int(crop_y+offset),int(crop_x-offset):int(crop_x+offset)]
                    if save_distance_cdc:
                        distance_crop = distance_map[int(crop_y-offset):int(crop_y+offset),int(crop_x-offset):int(crop_x+offset)]
                
                    type_path_crop = os.path.join(output_type_path, f"{(subject_id * iterations + iteration):06}.npz")

                    if save_distance_cdc:
                        np.savez(type_path_crop, image=image_crop, label=label_crop, distance=distance_crop)
                    else:
                        np.savez(type_path_crop, image=image_crop, label=label_crop)
    else:
        print(f"Dataset {output_path} has already been created")

def create_fn_dataset(input_path, output_path, split=0.8, radius=32):
    if not os.path.exists(output_path):
        subjects = sorted(os.listdir(input_path))
        n = len(subjects)
        types = ["train", "test"]

        counter = {
            "train": 0,
            "test": 0
        }

        for type in types:
            os.makedirs(os.path.join(output_path, type)) 
            for subject in tqdm(subjects):
                if subject == "stats_whole_image.npz" or subject == "stats_cdc.npz":
                    continue

                file_path = os.path.join(input_path, subject)
                data = np.load(file_path)

                image = data["image"]
                dt = data["prediction"]
                fp = data["fp"]
                fn = data["fn"]

                fn_indices = np.random.permutation(len(fn))
                fp_indices = np.random.permutation(len(fp))

                fn_indices = {
                    "train": fn_indices[:int(split*len(fn))],
                    "test": fn_indices[int(split*len(fn)):]
                }
                fp_indices = {
                    "train": fp_indices[:int(split*len(fp))],
                    "test": fp_indices[int(split*len(fp)):]
                }

                for index in fn_indices[type]:
                    y, x = np.round(fn[index,:]).astype(int)

                    image_crop = image[y-radius:y+radius+1,x-radius:x+radius+1]
                    dt_crop = dt[y-radius:y+radius+1,x-radius:x+radius+1]
                    classes = np.array([1, 0]) # Is FN

                    npz_path = os.path.join(output_path, type, f"{counter[type]:06d}.npz")
                    np.savez(npz_path, image=image_crop, dt=dt_crop, classes=classes)
                    counter[type] += 1

                for index in fp_indices[type]:
                    y, x = np.round(fp[index,:]).astype(int)

                    image_crop = image[y-radius:y+radius+1,x-radius:x+radius+1]
                    dt_crop = dt[y-radius:y+radius+1,x-radius:x+radius+1]
                    classes = np.array([0, 1]) # Is FP = hopefully TN

                    npz_path = os.path.join(output_path, type, f"{counter[type]:06d}.npz")
                    np.savez(npz_path, image=image_crop, dt=dt_crop, classes=classes)
                    counter[type] += 1


@DeprecationWarning
def generate_dataset_offline(input_path, 
    output_path, 
    cross_validation=True,
    crop_size=(256,256), 
    iterations=100,
    tile_uniformly=False, 
    margin=0,
    convert_to_dt=False,
    for_reference=False,
    erode_images=False,
    crop_mode=None):
    """
    Use this class to iterate a directory of (images, ground_truth)
    which shall be processed offline to be used in a training /
    validation / test dataset
    """

    if not os.path.exists(output_path):
        # Set available image types
        types = ["test", "train"] if cross_validation else ["validation", "test", "train"]

        # Get crop transform
        mode = "uniform_rectangle" if crop_mode == None else crop_mode
        crop_transform = CropTransform(crop_size[0], crop_size[1], margin=margin, mode=mode)

        # Iterate types
        for type in types:
            ct = 0
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

                if tile_uniformly:
                    # Create tiles of size crop_size
                    dy, dx = image.shape[0] % crop_size[0], image.shape[1] % crop_size[1]
                    dy_top, dy_bottom = int(np.floor(dy / 2)), int(np.ceil(dy / 2))
                    dx_left, dx_right = int(np.floor(dx / 2)), int(np.ceil(dx / 2))
                    image = image[dy_top:-dy_bottom,dx_left:-dx_right]
                    label = label[dy_top:-dy_bottom,dx_left:-dx_right]
                    ny, nx = int(image.shape[0] / crop_size[0]), int(image.shape[1] / crop_size[1])
                    images = np.array(np.split(np.array(np.split(image, ny, axis=0)), nx, axis=2)).reshape((-1,crop_size[0],crop_size[1]))
                    labels = np.array(np.split(np.array(np.split(label, ny, axis=0)), nx, axis=2)).reshape((-1,crop_size[0],crop_size[1]))

                    for id in range(ny * nx):
                        image_crop = images[id]
                        label_crop = labels[id]
                        type_path_crop = os.path.join(type_path, f"{(ct + id):06}.npz")
                        np.savez(type_path_crop, image=image_crop, label=label_crop, background_probability=background_probability, cone_probability=cone_probability)
                    ct += ny * nx
                else:
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

@DeprecationWarning
def convert_images_to_npz(
    input_path, 
    output_path,
    convert_to_dt=False,
    separator="\t",
    pad=56):
    """
    Convert a set of images (.tif) and cone locations (.csv)
    to our data format (.npz)
    """

    if not os.path.exists(output_path):
        # Get available subjects
        image_paths = sorted(glob.glob(os.path.join(input_path, "Images", "*"))) 
        cones_paths = sorted(glob.glob(os.path.join(input_path, "Cones", "*"))) 
        pairs = zip(image_paths, cones_paths)

        # Create output directory
        os.makedirs(output_path)

        # Iterate subjects and create npz files
        for idx, (image_path, cones_path) in enumerate(pairs):           
            # Read image
            image = imread(image_path, as_gray=True)

            # Read label
            label_indices = np.round(np.genfromtxt(cones_path, delimiter=separator)).astype(int) - 1
            image = np.pad(image, ((15,15),(15,15))) # Avoid indexing issues with cone locations; image is now of shape (180,180)
            label = np.zeros(image.shape)
            label[label_indices[:,1], label_indices[:,0]] = 1 # labels are in (x,y) format instead of (y,x)!

            # Clip image + label to size used in other papers (144,144)
            image = image[18:-18,18:-18]
            label = label[18:-18,18:-18]

            # Clip to size divisible by 8
            image, label = clip_image(image), clip_image(label)

            # Pad to (256,256)
            #image, label = np.pad(image,((pad,pad),(pad,pad))), np.pad(label,((pad,pad),(pad,pad)))

            # Image stats
            background_probability = np.sum(label == 0)
            cone_probability = np.sum(label == 1)
            background_probability, cone_probability = background_probability / (background_probability + cone_probability), cone_probability / (background_probability + cone_probability)

            # Optionally convert label to DT representation
            if convert_to_dt:
                label = class_to_dt(label)

            # Save .npz file
            npz_path = os.path.join(output_path, f"{idx:06d}.npz")
            np.savez(npz_path, image=image, label=label, background_probability=background_probability, cone_probability=cone_probability) 
    else:
        print(f"Dataset {output_path} has already been generated")
