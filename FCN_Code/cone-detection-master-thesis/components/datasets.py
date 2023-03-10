from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as TF
from torchvision.transforms import GaussianBlur
import random
import os
import numpy as np
import torch
from sklearn.model_selection import KFold

from utils.image import class_to_dt, dt_to_class
from utils.transforms import RotationTransform, CropTransform, CustomElasticTransform

class ConesDataset(Dataset):
    """
    Use this class to represent a dataset of AOSLO images
    """

    def __init__(self, path, train, convert_to_dt=False, pre_generated=False):
        super(ConesDataset, self).__init__()

        # Set dataset properties
        self.path = path
        self.train = train
        self.convert_to_dt = convert_to_dt
        self.pre_generated = pre_generated
        self.images = []
        self.identifiers = []
        self.labels = []
        self.cdc20 = []
        self.background_probability = 1
        self.cone_probability = 0

        # Transforms
        self.crop = CropTransform(256, 256)
        self.rotate = RotationTransform(angles=[0,90,180,270])
        self.blur = GaussianBlur(kernel_size=(5,5), sigma=(0.1,5))
        self.elastic = CustomElasticTransform(alpha=20.0, sigma=3.0)

        # Get files
        files = sorted(os.listdir(path))
        self.dataset_len = len(files)

        if self.pre_generated:
            random_index = random.randint(0, self.dataset_len - 1)
            npz = np.load(os.path.join(path, files[random_index]))
            self.background_probability = npz["background_probability"]
            self.cone_probability = npz["cone_probability"]
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

                # Convert to DT and potentially remove black stitching background
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
        else:
            npz = np.load(os.path.join(self.path, f"{idx:06}.npz"))
            img = npz["image"] / 255.0 #uint8 image
            label = npz["label"]
            if self.convert_to_dt:
                label = class_to_dt(label)
            img, label = torch.from_numpy(img).unsqueeze(0), torch.from_numpy(label).unsqueeze(0)
            identifier = idx

        # Pre generated datasets already consist of crops
        if not self.pre_generated:
            # On training, get a random 256x256 crop of the image
            if self.train:
                img, label = self.crop(img, label)

        # On training, do some transformations
        if self.train:
            # Random rotation
            img, label = self.rotate(img, label)

            # Random blur
            #img = self.blur(img)

            # Random elastic distortion
            #img, label = self.elastic(img, label)

        # Always apply auto contrast
        img = TF.autocontrast(img)

        return img, label, identifier

    def get_dataloader(self, shuffle=True, batch_size=64):
        """
        Get a DataLoader for this dataset with the given batch_size
        and shuffle behaviour
        """
        return DataLoader(self, batch_size=batch_size, shuffle=shuffle)

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