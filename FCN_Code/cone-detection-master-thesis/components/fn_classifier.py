import torch
import torch.nn as nn
import torch.nn.functional as func
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import os
from datetime import datetime
from tqdm import tqdm

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

class FNDataset(Dataset):
    """
    Dataset of FN
    """

    def __init__(self, path, train=False):
        super(FNDataset, self).__init__()

        # Set dataset properties
        self.path = path
        self.train = train

        # Transforms
        self.rotate = RotationTransform(angles=[0,90,180,270])

        # Get files
        files = sorted(os.listdir(path))
        self.dataset_len = len(files)

    def __len__(self):
        return self.dataset_len

    def __getitem__(self, idx):
        data = np.load(os.path.join(self.path, f"{idx:06d}.npz"))
        image = data["image"]
        dt = data["dt"]
        classes = data["classes"]

        image = torch.from_numpy(image).unsqueeze(0)
        dt = torch.from_numpy(dt).unsqueeze(0)
        classes = torch.from_numpy(classes).unsqueeze(0)

        if self.train:
            image, dt = self.rotate(image, dt)

        return image, dt, classes

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

class FNClassifier(nn.Module):
    """
    A classifier for false negatives
    """

    def __init__(self, train_dataloader, validation_dataloader=None, batch_size=32):
        super(FNClassifier, self).__init__()

        # Props
        self.lr = 3e-4 #0.001
        self.train_dataloader = train_dataloader
        self.validation_dataloader = validation_dataloader
        self.batch_size = batch_size

        # Select device
        use_cuda = torch.cuda.is_available()
        self.device = torch.device("cuda:0" if use_cuda else 'cpu')

        # Build architecture
        self.conv1 = nn.LazyConv2d(16, (3,3), padding=(1,1))
        self.norm1 = nn.BatchNorm2d(16)

        self.conv2 = nn.LazyConv2d(1, (3,3), padding=(1,1))
        self.norm2 = nn.BatchNorm2d(1)

        #self.conv3 = nn.LazyConv2d(1, (3,3), padding=(1,1))
        #self.norm3 = nn.BatchNorm2d(1)

        self.fc1 = nn.Linear(65*65, 128)
        self.fc2 = nn.Linear(128, 2)
        self.softmax = nn.Softmax(dim=2)

        # Move model to preferred device
        self.to(self.device)

        # Criterion and Optimizer          
        self.criterion = nn.BCELoss()
        self.optimizer = optim.Adam(self.parameters(), lr=self.lr)

    def forward(self, x):
        x = self.conv1(x)
        x = self.norm1(x)
        #print(x.shape)

        x = self.conv2(x)
        x = self.norm2(x)
        #print(x.shape)

        #x = self.conv3(x)
        #x = self.norm3(x)
        #print(x.shape)

        x = torch.flatten(x, start_dim=2)
        #print(x.shape)

        x = self.fc1(x)
        x = self.fc2(x)
        #print(x.shape)
        x = self.softmax(x)
        #print(x.shape)

        return x
    
    def save(self, path, title):
        """ 
        Save the net 
        """
        file_path = os.path.join(path, f"{title}.pth")
        torch.save(self.state_dict(), file_path)
    
    def load(self,filepath):
        """ 
        Load the state of a trained net 
        """
        if torch.cuda.is_available():
            self.load_state_dict(torch.load(filepath))
        else:
            self.load_state_dict(torch.load(filepath, map_location=torch.device('cpu')))

    def _get_parameter_count(self, only_trainable=False):
        return sum(p.numel() for p in self.parameters() if not only_trainable or p.requires_grad)
    
    def train_network(self, max_epochs=100, run_id=None, verbose=True, early_stopping=False):
        # Tensorboard SummaryWriter
        now = datetime.now()
        stamp = now.strftime("%Y%m%d-%H%M")
        if run_id is not None:
            stamp = f"{stamp}-{run_id}"
        writer = SummaryWriter('../runs/{}'.format(stamp))

        # Logging variables
        min_training_loss = 1e10
        min_validation_loss = 1e10

        # Train model for n epochs
        iterator = range(max_epochs) if verbose else tqdm(range(max_epochs))
        for current_epoch in iterator: #range(epochs):
            running_loss = 0.0
            for data in self.train_dataloader:
                images, dts, classes = data

                self.optimizer.zero_grad()

                images, dts, classes = images.to(self.device).float(), dts.to(self.device).float(), classes.to(self.device).float()

                input = torch.cat([images, dts], dim=1)
                y = self(input)

                #classes = torch.cat([labels, 1-labels], dim=1)
                loss = self.criterion(y, classes)
                loss.backward()

                self.optimizer.step()
                running_loss += loss.item()

                # del imgs, labels, loss
                #torch.cuda.empty_cache()

            # Log performance
            avg_run_loss = running_loss / len(self.train_dataloader) 
            if verbose:
                print("Average running TRAINING loss for epoch {}: {}".format(current_epoch + 1, avg_run_loss))

            if avg_run_loss < min_training_loss:
                min_training_loss = avg_run_loss

            writer.add_scalar('training loss',
                avg_run_loss,
                (current_epoch + 1) * len(self.train_dataloader)) 

            # If a validation DataLoader was given, validate the network
            running_loss = 0.0
            if self.validation_dataloader is not None:
                self.eval()
                with torch.no_grad():
                    for data in self.validation_dataloader:
                        images, dts, classes = data                  

                        images, dts, classes = images.to(self.device).float(), dts.to(self.device).float(), classes.to(self.device).float()
                        
                        input = torch.cat([images, dts], dim=1)
                        y = self(input)

                        # classes = torch.cat([labels, 1-labels], dim=1)
                        loss = self.criterion(y, classes)
                        running_loss += loss.item()

                    avg_run_loss = running_loss / len(self.validation_dataloader)

                    if verbose:
                        print("Average running VALIDATION loss for epoch {}: {}".format(current_epoch + 1, avg_run_loss))

                    if avg_run_loss < min_validation_loss:
                        min_validation_loss = avg_run_loss
                        if run_id is not None:
                            self.save("../nets", stamp)
                        else:
                            self.save("../nets", f"fn_classifier_{(current_epoch+1):03}_loss_{min_validation_loss}")
                    elif early_stopping and avg_run_loss > 1.2 * min_validation_loss:
                        # Stop if the validation loss does not improve
                        return (current_epoch+1), min_training_loss, min_validation_loss

                    writer.add_scalar('validation loss',
                        avg_run_loss,
                        (current_epoch + 1) * len(self.validation_dataloader))
                self.train()

        # Clean up after training
        if self.validation_dataloader is None:
            self.save("../nets", f"fn_classifier_{(current_epoch+1):03}_loss_{min_training_loss}")
        writer.close()

        return (current_epoch+1), min_training_loss, min_validation_loss

    @staticmethod
    def train_network_single_cross_validation(fold_id, train_dataloader, validation_dataloader, max_epochs=100, early_stopping=False):
        """
        Train an instance of FNClassifier using cross validation 
        for at most the given number of epochs with a single fold.

        Optionally use early stopping.
        """
        model = FNClassifier(train_dataloader, validation_dataloader=validation_dataloader, batch_size=32)
            
        trained_epochs, min_training_loss, min_validation_loss = model.train_network(
            max_epochs=max_epochs, 
            run_id=f"fold-{fold_id}", 
            verbose=True, #False, 
            early_stopping=early_stopping)

        print(f"Fold {fold_id}: {trained_epochs} epochs, ({min_training_loss},{min_validation_loss})")
        del model
        torch.cuda.empty_cache()

    @staticmethod
    def train_networks_k_fold_cross_validation(dataloaders, max_epochs=100, early_stopping=False):
        """
        Train instances of FNClassifier using k-fold cross validation 
        for at most the given number of epochs on each fold.

        Optionally use early stopping.
        """

        for k, (train_dataloader, validation_dataloader) in enumerate(dataloaders):
            model = FNClassifier(train_dataloader, validation_dataloader=validation_dataloader, batch_size=32)
            
            trained_epochs, min_training_loss, min_validation_loss = model.train_network(
                max_epochs=max_epochs, 
                run_id=f"fold-{k}", 
                verbose=True, #False, 
                early_stopping=early_stopping)

            print(f"Fold {k}: {trained_epochs} epochs, ({min_training_loss},{min_validation_loss})")
            del model
            torch.cuda.empty_cache()