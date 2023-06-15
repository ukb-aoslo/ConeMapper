from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as TF
from torchvision.transforms import GaussianBlur
import random
import os
import numpy as np
import torch
from sklearn.model_selection import KFold

import scipy.ndimage as ndi
from skimage import exposure
from skimage.morphology import disk
from skimage.filters import rank

from utils.image import class_to_dt, dt_to_class, class_to_gaussians, mask_array
from utils.transforms import RotationTransform, CropTransform, RandomElasticTransform

class ConesDataset(Dataset):
    """
    Use this class to represent a dataset of AOSLO images
    """

    def __init__(self, 
        path, 
        train, 
        convert_to_dt=False,
        convert_to_gaussians=False, 
        pre_generated=False,
        augment_rotate=False,
        augment_blur=False,
        augment_elastic=False,
        dilation=0,
        adaptive_dilation=False,
        adaptive_dilation_interval=30):
        super(ConesDataset, self).__init__()

        # Set dataset properties
        self.path = path
        self.train = train
        self.convert_to_dt = convert_to_dt
        self.convert_to_gaussians = convert_to_gaussians
        self.pre_generated = pre_generated
        self.augment_rotate = augment_rotate
        self.augment_blur = augment_blur
        self.augment_elastic = augment_elastic
        self.dilation = dilation
        self.adaptive_dilation = adaptive_dilation
        self.adaptive_dilation_interval = adaptive_dilation_interval

        self.images = []
        self.identifiers = []
        self.labels = []
        self.cdc20 = []
        self.exact_gt = []
        self.background_probability = 1
        self.cone_probability = 0

        # Transforms
        self.crop = CropTransform(256, 256)
        self.rotate = RotationTransform(angles=[0,90,180,270])
        self.blur = GaussianBlur(kernel_size=(3,3), sigma=1.0/3.0) #(5,5), sigma=(0.1,1.0))
        self.elastic = RandomElasticTransform(max_alpha=4.0, sigma=1.0)

        # Get files
        files = sorted(os.listdir(path))
        self.dataset_len = len(files)

        if self.pre_generated:
            random_index = random.randint(0, self.dataset_len - 1)
            npz = np.load(os.path.join(path, files[random_index]))
            # self.background_probability = npz["background_probability"]
            # self.cone_probability = npz["cone_probability"]
        else:
            # Read all .npz files in the given path
            for file in files:
                self.identifiers.append(file.replace(".npz", ""))
                npz = np.load(os.path.join(path, file))

                # Get image
                image = npz["image"] / 255.0 #uint8 image
                self.images.append(torch.Tensor(image).unsqueeze(0))

                # Get label
                label = npz["label"]

                self.background_probability += np.sum(label == 0)
                self.cone_probability += np.sum(label == 1)
                self.exact_gt.append(npz["exact_gt"])

                if self.dilation > 0:
                    label = ndi.binary_dilation(label, iterations=self.dilation)

                # Convert to DT
                if self.convert_to_dt:
                    label = class_to_dt(label)

                self.labels.append(torch.Tensor(label).unsqueeze(0))

                # Get cone density centroid location
                self.cdc20.append(npz["cdc20"])   

            # Compute class probabilities
            self.background_probability, self.cone_probability = self.background_probability / (self.background_probability + self.cone_probability), self.cone_probability / (self.background_probability + self.cone_probability)    

    def __len__(self):
        return self.dataset_len

    def __getitem__(self, idx):
        # Get image and label pair
        if not self.pre_generated:
            img = self.images[idx]
            label = self.labels[idx]
            identifier = self.identifiers[idx]
            cdc = self.cdc20[idx].astype(int)
            exact_gt = self.exact_gt[idx]
        else:
            npz = np.load(os.path.join(self.path, f"{idx:06}.npz"))
            img = npz["image"] / 255.0 #uint8 image
            label = npz["label"]
            distance = npz["distance"]

            if self.adaptive_dilation:
                pixel_res = 600 # Specific to UKB dataset
                scale = self.adaptive_dilation_interval

                new_label = np.zeros_like(label)
                max_distance = np.max(distance)
                upper_bound = int(np.ceil(max_distance / pixel_res * 60 / scale))

                for dilation_factor in range(1,upper_bound+1):
                    min_d = (dilation_factor-1) * scale / 60 * pixel_res
                    max_d = dilation_factor * scale / 60 * pixel_res
                    masked = label * ((min_d <= distance).astype(int) & (distance <= max_d).astype(int))
                    masked = ndi.binary_dilation(masked, iterations=dilation_factor)
                    new_label += masked

                label = new_label

            if self.dilation > 0:
                label = ndi.binary_dilation(label, iterations=self.dilation)

            if self.convert_to_dt:
                label = class_to_dt(label)

            if self.convert_to_gaussians:
                label = class_to_gaussians(label) #, 1.25 - 0.5 * img)

            img, label = torch.from_numpy(img).unsqueeze(0), torch.from_numpy(label).unsqueeze(0)
            identifier = idx

        # Pre generated datasets already consist of crops
        if not self.pre_generated:
            # On training, get a random 256x256 crop of the image
            if self.train:
                img, label = self.crop(img, label)

        # On training, do some transformations
        if self.train:
            if self.augment_rotate:
                # Random rotation
                img, label = self.rotate(img, label)

            if self.augment_blur:
                # Random blur
                img = self.blur(img)

            if self.augment_elastic:
                # Random elastic distortion
                img, label = self.elastic(img, label)

        # Always apply auto contrast
        # NO! THIS HAS A NEGATIVE EFFECT AT EVALUATION TIME
        # AS IMAGES WILL NOT HAVE ENOUGH INTENSITY TO TRIGGER CERTAIN
        # FEATURE MAPS
        # img = TF.autocontrast(img)

        # Let's not do this: Instead use local histogram equalization
        # if self.train:
        #     # BUT: Do randomly brighten or darken the image
        #     bf = 1.0 + 0.08 * np.random.randn() # most values in range [0.75,1.25]
        #     img = TF.adjust_brightness(img, bf)

        if self.pre_generated:
            return img, label, identifier, [], distance, []
        else:
            return img, label, identifier, cdc, [], exact_gt

    def get_dataloader(self, shuffle=True, batch_size=64):
        """
        Get a DataLoader for this dataset with the given batch_size
        and shuffle behaviour
        """
        return DataLoader(self, batch_size=batch_size, shuffle=shuffle)

    def get_k_fold_splits(self, k, path, shuffle=True):
        """
        Get k splits of the dataset into (training, validation)
        and save them as npz files
        """
        kfold = KFold(n_splits=k, shuffle=shuffle)

        if not os.path.exists(path):
            os.makedirs(path)

        for fold, (train_ids, test_ids) in enumerate(kfold.split(self)):
            file_path = os.path.join(path, f"fold_{fold}.npz")
            np.savez(file_path, train_ids=train_ids, test_ids=test_ids)

    def get_single_dataloader_cross_validation(self, fold_file_path, batch_size=64):
        """
        Get two dataloaders with the given batch_size from train_ids and
        test_ids stored in fold_file_path (.npz)
        """
        fold = np.load(fold_file_path)
        train_ids, test_ids = fold["train_ids"], fold["test_ids"]

        # Define subsamplers
        train_subsampler = torch.utils.data.SubsetRandomSampler(train_ids)
        test_subsampler = torch.utils.data.SubsetRandomSampler(test_ids)
        
        # Define data loaders for training and testing data in this fold
        trainloader = torch.utils.data.DataLoader(self, batch_size=batch_size, sampler=train_subsampler)
        testloader = torch.utils.data.DataLoader(self,batch_size=batch_size, sampler=test_subsampler)

        return trainloader, testloader

    def get_dataloaders_k_fold_cross_validation(self, k, shuffle=True, batch_size=64):
        """
        Get a list of DataLoaders for this dataset with the given batch_size 
        and shuffle behaviour following k-fold cross validation
        """
        # For a useful guide, see:
        # https://github.com/christianversloot/machine-learning-articles/blob/main/how-to-use-k-fold-cross-validation-with-pytorch.md
        kfold = KFold(n_splits=k, shuffle=shuffle)
        dataloaders = []

        for fold, (train_ids, test_ids) in enumerate(kfold.split(self)):
            # Define subsamplers
            train_subsampler = torch.utils.data.SubsetRandomSampler(train_ids)
            test_subsampler = torch.utils.data.SubsetRandomSampler(test_ids)
            
            # Define data loaders for training and testing data in this fold
            trainloader = torch.utils.data.DataLoader(self, batch_size=batch_size, sampler=train_subsampler)
            testloader = torch.utils.data.DataLoader(self,batch_size=batch_size, sampler=test_subsampler)

            # Append to list
            dataloaders.append((trainloader, testloader))

        return dataloaders