import torch
import torch.nn as nn
import torch.nn.functional as func
import torch.optim as optim
import numpy as np
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
import os

from tqdm import tqdm

from torchvision import models

#from components.postprocessing import evaluate, coordinates_to_mask
from components.loss import DistanceAwareFocalLoss, MaskedMSELoss

from utils.cdc import arcmin_to_pixels, get_circular_mask, get_maximum_extent, pixels_to_arcmin
from utils.image import clip_image
from utils.transforms import LabelRegularityTransform, DistanceTransform

class FCN(nn.Module):
    """
    Fully Convolutional Network as described by 

    'Automatic Detection of Cone Photoreceptors With Fully Convolutional Networks' (Hamwood et al., 2019)
    """

    def __init__(self, train_dataloader, validation_dataloader=None, test_dataloader=None, lr=0.001, depth=3, input_channels=1, initial_feature_maps=16, use_class_weights=False,
                blocks_per_resolution_layer=1, use_MSE_loss=False, masked_MSE_loss=False, use_distance_aware_focal_loss=False, regularity_aware_training=False,
                use_distance_information=False, use_refiner=False, use_lr_scheduler=False):
        super(FCN, self).__init__()

        # Set properties
        self.depth = depth
        # self.scaling_factor = 2
        self.feature_maps_factor = 2
        self.input_channels = input_channels
        self.initial_feature_maps = initial_feature_maps
        self.lr = lr
        self.train_dataloader = train_dataloader
        self.validation_dataloader = validation_dataloader
        self.test_dataloader = test_dataloader
        self.use_MSE_loss = use_MSE_loss
        self.masked_MSE_loss = masked_MSE_loss
        self.use_distance_aware_focal_loss = use_distance_aware_focal_loss
        self.use_class_weights = use_class_weights
        self.regularity_aware_training = regularity_aware_training
        self.use_distance_information = use_distance_information
        self.use_refiner = use_refiner
        self.use_lr_scheduler = use_lr_scheduler

        # Set some properties based on other properties
        if self.regularity_aware_training:
            self.input_channels = 2
            self.label_modifier = LabelRegularityTransform(tpr=0.98, fdr=0.005, cone_probability=0.02)
            self.dt_modifier = DistanceTransform()

        if self.use_distance_information:
            self.input_channels = 2

        # Computed parameters
        self.feature_maps = [self.input_channels]
        self.feature_maps.extend([int(self.initial_feature_maps * self.feature_maps_factor ** d) for d in range(0,self.depth+1)])
        self.skip_connections_values = []

        # Select device
        use_cuda = torch.cuda.is_available()
        self.device = torch.device("cuda:0" if use_cuda else 'cpu')

        # Parts of the network
        self.encoder = nn.ModuleList()
        self.skip_connections = nn.ModuleList()
        self.center_blocks = nn.ModuleList()
        self.decoder = nn.ModuleList()
        self.refiner = nn.ModuleList()
        self.predictor = nn.ModuleList()
        self.transfer_model = nn.ModuleList()
        
        # Add blocks to the encoder
        for i in range(0, self.depth):
            self.encoder.append(nn.Conv2d(self.feature_maps[i], self.feature_maps[i+1], kernel_size=(3,3), padding=(1,1)))
            self.encoder.append(nn.BatchNorm2d(self.feature_maps[i+1]))
            self.encoder.append(nn.ReLU())

            for b in range(blocks_per_resolution_layer-1):
                self.encoder.append(nn.Conv2d(self.feature_maps[i+1], self.feature_maps[i+1], kernel_size=(3,3), padding=(1,1)))
                self.encoder.append(nn.BatchNorm2d(self.feature_maps[i+1]))
                self.encoder.append(nn.ReLU())

            self.encoder.append(nn.MaxPool2d((2,2)))

        # Create skip connections
        for i in range(0, self.depth):
            self.skip_connections.append(nn.Identity())

        # Add center blocks
        self.center_blocks.append(nn.Conv2d(self.feature_maps[-2], self.feature_maps[-1], kernel_size=(3,3), padding=(1,1)))
        self.center_blocks.append(nn.BatchNorm2d(self.feature_maps[-1]))
        self.center_blocks.append(nn.ReLU())

        for b in range(blocks_per_resolution_layer-1):
            self.center_blocks.append(nn.Conv2d(self.feature_maps[-1], self.feature_maps[-1], kernel_size=(3,3), padding=(1,1)))
            self.center_blocks.append(nn.BatchNorm2d(self.feature_maps[-1]))
            self.center_blocks.append(nn.ReLU())

        self.center_blocks.append(nn.Dropout2d(p=0.2 ))#5)) #p=0.2))

        # Add blocks to the decoder
        for i in range(self.depth + 1, 1, -1):
            # They claim to use 3x3 ConvTranspose2d, however the paper they cite regarding the net 
            # architecture 'U-Net - Convolutional Networks for Biomedical Image Segmentation' uses
            # 2x2 ConvTranspose2d
            #self.decoder.append(nn.ConvTranspose2d(self.feature_maps[i], self.feature_maps[i-1], kernel_size=(2,2), stride=(2,2)))

            # In their implementation, they use 4x4 transposed convolution though :/
            self.decoder.append(nn.ConvTranspose2d(self.feature_maps[i], self.feature_maps[i-1], kernel_size=(4,4), stride=(2,2), padding=(1,1)))
            
            # Skip connections double the number of input feature_maps here!
            self.decoder.append(nn.Conv2d(self.feature_maps[i], self.feature_maps[i-1], kernel_size=(3,3), padding=(1,1)))
            self.decoder.append(nn.BatchNorm2d(self.feature_maps[i-1]))
            self.decoder.append(nn.ReLU())

            for b in range(blocks_per_resolution_layer-1):
                self.decoder.append(nn.Conv2d(self.feature_maps[i-1], self.feature_maps[i-1], kernel_size=(3,3), padding=(1,1)))
                self.decoder.append(nn.BatchNorm2d(self.feature_maps[i-1]))
                self.decoder.append(nn.ReLU())

        # Add some more layers to predictor if necessary
        if self.use_refiner:
            # Pyramidal Convolution - https://arxiv.org/abs/2006.11538
            self.refiner.append(nn.Conv2d(self.feature_maps[1], self.feature_maps[1], (1,1)))
            self.refiner.append(nn.BatchNorm2d(self.feature_maps[1]))
            self.refiner.append(nn.ReLU())

            self.refiner.append(nn.Conv2d(self.feature_maps[1], self.feature_maps[1], (9,9), padding=(4,4)))
            self.refiner.append(nn.Conv2d(self.feature_maps[1], self.feature_maps[1], (7,7), padding=(3,3)))
            self.refiner.append(nn.Conv2d(self.feature_maps[1], self.feature_maps[1], (5,5), padding=(2,2)))
            self.refiner.append(nn.Conv2d(self.feature_maps[1], self.feature_maps[1], (3,3), padding=(1,1)))

            self.refiner.append(nn.BatchNorm2d(self.feature_maps[1] * 4))
            self.refiner.append(nn.ReLU())
            self.refiner.append(nn.Conv2d(self.feature_maps[1] * 4, self.feature_maps[1], (1,1)))

        # Add blocks to the predictor
        self.predictor.append(nn.Conv2d(self.feature_maps[1], 2, (1,1)))

        # Last softmax output layer
        if self.use_MSE_loss:
            self.predictor.append(nn.Conv2d(2, 1, (1,1)))
            #self.predictor.append(nn.ReLU())
        else:
            self.predictor.append(nn.Softmax2d())

        # Move model to preferred device
        self.to(self.device)

        # Criterion and Optimizer           
        if self.use_MSE_loss:
            if self.masked_MSE_loss:
                self.criterion = MaskedMSELoss(better_mask=True)
            else:
                self.criterion = nn.MSELoss()
            #self.criterion = nn.HuberLoss(delta=1.0) 
        else:
            if self.use_distance_aware_focal_loss:
                self.criterion = DistanceAwareFocalLoss(alpha=1.0, beta=1.0)
            else:
                ds = self.train_dataloader.dataset
                class_weights = torch.Tensor([1.0 / ds.cone_probability, 1.0 / ds.background_probability]) if use_class_weights else None
                if class_weights != None:
                    class_weights = class_weights.to(self.device)
                self.criterion = nn.CrossEntropyLoss(weight=class_weights, reduction='mean') #FCN.custom_loss #nn.CrossEntropyLoss()  #nn.MSELoss()          

        self.optimizer = optim.Adam(self.parameters(), lr=self.lr)
        if self.use_lr_scheduler:
            #self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode="min", patience=5)
            self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=50, gamma=0.5)

    def forward(self, x):
        """
        Forward a tensor (batch size, feature maps, width, height) input through the FCN
        """

        # Encode
        slip_depth = 0
        for module in self.encoder:
            #print(x.shape)
            if type(module) is nn.MaxPool2d:
                self.skip_connections_values.append(self.skip_connections[slip_depth](x.clone()))
                slip_depth += 1
            x = module(x)

        # Apply center blocks
        for module in self.center_blocks:
            #print(x.shape)
            x = module(x)

        # Decode
        slip_depth = self.depth - 1
        for module in self.decoder:
            #print(x.shape)
            x = module(x)
            #print(x.shape)
            if type(module) == nn.ConvTranspose2d:
                #print(self.skip_connections_values[slip_depth].shape)
                x = torch.cat([x, self.skip_connections_values[slip_depth]], dim=1)
                slip_depth -= 1
        
        # Refiner (Pyramid convolution)
        if self.use_refiner:
            for idx in range(3):
                x = self.refiner[idx](x)
            refined = [self.refiner[idx](x) for idx in range(3,7)]
            x = torch.cat(refined, dim=1)
            for idx in range(7,10):
                x = self.refiner[idx](x)

        # Predict
        for module in self.predictor:
            #print(x.shape)
            x = module(x)

        # Clear skip connections
        self.skip_connections_values.clear()

        return x

    def prepare_data(self,data):
        """
        Prepare the input data
        - create a GPU tensor from numpy arrays
        - move CPU tensors to GPU
        """
        if type(data) is np.ndarray:
            tensor = torch.from_numpy(data).float().to(self.device)
            if len(tensor.shape) < 4:
                tensor = tensor.unsqueeze(1)
            return tensor
        elif type(data) is torch.Tensor:
            tensor = data.float().to(self.device)
            if len(tensor.shape) < 4:
                tensor = tensor.unsqueeze(1)
            return tensor
        else:
            print("Unknown data type", type(data))
            return None

    def save(self,path,title):
        """ 
        Save the net 
        """
        #now = datetime.now()
        #stamp = now.strftime("%Y%m%d-%H%M")
        #torch.save(self.state_dict(), "{}{}{}-{}.pth".format(path,"" if path == "" else "/",title,stamp))
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
        """
        Train this instance of a FCN for the given number
        of epochs 
        """

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
                imgs, labels, identifiers, cdc, distances, exact_gt = data

                self.optimizer.zero_grad()

                if self.regularity_aware_training:
                    modified_labels = self.label_modifier((labels == 0).float()) # REMEMBER: Label is DT!
                    modified_labels = self.dt_modifier(modified_labels)

                    imgs, labels = self.prepare_data(imgs), self.prepare_data(labels) # i.e. move to GPU in this case
                    modified_labels = self.prepare_data(modified_labels)
                    inputs = torch.cat([imgs, modified_labels], dim=1)
                elif self.use_distance_information:
                    imgs, labels, distances = self.prepare_data(imgs), self.prepare_data(labels), self.prepare_data(distances)
                    # print(imgs.shape, distances.shape)
                    inputs = torch.cat([imgs, distances], dim=1)
                else:
                    imgs, labels = self.prepare_data(imgs), self.prepare_data(labels) # i.e. move to GPU in this case
                    inputs = imgs

                y = self(inputs)

                if self.use_MSE_loss:
                    if self.masked_MSE_loss:
                        if self.use_class_weights:
                            loss = self.criterion(imgs, y, labels, 
                                background_probability=self.train_dataloader.dataset.background_probability,
                                cone_probability=self.train_dataloader.dataset.cone_probability)
                        else:
                            loss = self.criterion(imgs, y, labels)
                    else:
                        loss = self.criterion(y, labels)                 
                else:
                    loss = self.criterion(y, torch.cat([labels, 1-labels], dim=1))

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
                if self.validation_dataloader is None:
                    if run_id is not None:
                        self.save("../nets", stamp)
                    else:
                        self.save("../nets", f"fcn_epoch_{(current_epoch+1):03}_loss_{min_training_loss}")

            writer.add_scalar('training loss',
                avg_run_loss,
                (current_epoch + 1) * len(self.train_dataloader)) 

            # If a validation DataLoader was given, validate the network
            running_loss = 0.0
            if self.validation_dataloader is not None:
                self.eval()
                with torch.no_grad():
                    for data in self.validation_dataloader:
                        imgs, labels, identifiers, cdc, distances, exact_gt = data                     

                        if self.regularity_aware_training:
                            modified_labels = self.label_modifier((labels == 0).float()) # REMEMBER: Label is DT!
                            modified_labels = self.dt_modifier(modified_labels)

                            imgs, labels = self.prepare_data(imgs), self.prepare_data(labels) # i.e. move to GPU in this case
                            modified_labels = self.prepare_data(modified_labels)
                            inputs = torch.cat([imgs, modified_labels], dim=1)
                        elif self.use_distance_information:
                            imgs, labels, distances = self.prepare_data(imgs), self.prepare_data(labels), self.prepare_data(distances)
                            inputs = torch.cat([imgs, distances], dim=1)
                        else:
                            imgs, labels = self.prepare_data(imgs), self.prepare_data(labels) # i.e. move to GPU in this case
                            inputs = imgs

                        y = self(inputs)
                        # y = self(imgs)

                        if self.use_MSE_loss:
                            if self.masked_MSE_loss:
                                if self.use_class_weights:
                                    loss = self.criterion(imgs, y, labels, 
                                        background_probability=self.train_dataloader.dataset.background_probability,
                                        cone_probability=self.train_dataloader.dataset.cone_probability)
                                else:
                                    loss = self.criterion(imgs, y, labels)
                            else:
                                loss = self.criterion(y, labels)
                        else:
                            loss = self.criterion(y, torch.cat([labels, 1-labels], dim=1))
                        running_loss += loss.item()

                    avg_run_loss = running_loss / len(self.validation_dataloader)

                    if verbose:
                        print("Average running VALIDATION loss for epoch {}: {}".format(current_epoch + 1, avg_run_loss))

                    if self.use_lr_scheduler:
                        #self.scheduler.step(avg_run_loss)
                        self.scheduler.step()

                    if avg_run_loss < min_validation_loss:
                        min_validation_loss = avg_run_loss
                        if run_id is not None:
                            self.save("../nets", stamp)
                        else:
                            self.save("../nets", f"fcn_epoch_{(current_epoch+1):03}_loss_{min_validation_loss}")
                    elif early_stopping and avg_run_loss > 1.2 * min_validation_loss:
                        # Stop if the validation loss does not improve
                        return (current_epoch+1), min_training_loss, min_validation_loss

                    writer.add_scalar('validation loss',
                        avg_run_loss,
                        (current_epoch + 1) * len(self.validation_dataloader))
                self.train()

        # Clean up after training
        # if self.validation_dataloader is None:
        #     if run_id is not None:
        #         self.save("../nets", stamp)
        #     else:
        #         self.save("../nets", f"fcn_epoch_{(current_epoch+1):03}_loss_{min_training_loss}")
        #     #self.save("../nets", f"fcn_epoch_{(current_epoch+1):03}_loss_{min_training_loss}")
        writer.close()

        return (current_epoch+1), min_training_loss, min_validation_loss

    @staticmethod
    def train_network_single_cross_validation(fold_id, train_dataloader, validation_dataloader, max_epochs=100, lr=3e-4, early_stopping=False, regularity_aware_training=False,
                                              use_distance_information=False, use_refiner=False, use_lr_scheduler=False):
        """
        Train an instance of a FCN using cross validation 
        for at most the given number of epochs with a single fold.

        Optionally use early stopping.
        """
        model = FCN(train_dataloader, validation_dataloader=validation_dataloader, lr=lr, depth=3, input_channels=1, 
                initial_feature_maps=32, blocks_per_resolution_layer=2, use_MSE_loss=True, masked_MSE_loss=True, use_class_weights=False,
                regularity_aware_training=regularity_aware_training, use_distance_information=use_distance_information, use_refiner=use_refiner,
                use_lr_scheduler=use_lr_scheduler)
            
        trained_epochs, min_training_loss, min_validation_loss = model.train_network(
            max_epochs=max_epochs, 
            run_id=f"fold-{fold_id}", 
            verbose=False, 
            early_stopping=early_stopping)

        print(f"Fold {fold_id}: {trained_epochs} epochs, ({min_training_loss},{min_validation_loss})")
        del model
        torch.cuda.empty_cache()

    @staticmethod
    def train_networks_k_fold_cross_validation(dataloaders, max_epochs=100, lr=3e-4, early_stopping=False, regularity_aware_training=False, use_distance_information=False,
                                               use_refiner=False, use_lr_scheduler=False):
        """
        Train instances of a FCN using k-fold cross validation 
        for at most the given number of epochs on each fold.

        Optionally use early stopping.
        """

        for k, (train_dataloader, validation_dataloader) in enumerate(dataloaders):
            model = FCN(train_dataloader, validation_dataloader=validation_dataloader, lr=lr, depth=3, input_channels=1, 
                initial_feature_maps=32, blocks_per_resolution_layer=2, use_MSE_loss=True, masked_MSE_loss=True, use_class_weights=False,
                regularity_aware_training=regularity_aware_training, use_distance_information=use_distance_information, use_refiner=use_refiner,
                use_lr_scheduler=use_lr_scheduler)
            
            trained_epochs, min_training_loss, min_validation_loss = model.train_network(
                max_epochs=max_epochs, 
                run_id=f"fold-{k}", 
                verbose=False, 
                early_stopping=early_stopping)

            print(f"Fold {k}: {trained_epochs} epochs, ({min_training_loss},{min_validation_loss})")
            del model
            torch.cuda.empty_cache()

    # def hyperparameter_search(self, dist_criteria, thresholds, sigmas, max_hs, hamwoods_free_zone=False, distance_transform=False):
    #     """
    #     Perform hyperparameter search, e.g. for
    #     post-processing parameters
    #     """
        
    #     dataset_num = len(self.validation_dataloader.dataset)
    #     dices = np.zeros((len(dist_criteria), len(thresholds), len(sigmas), len(max_hs), dataset_num))

    #     self.eval()
    #     with torch.no_grad():
    #         for d_idx, dist_criterium in enumerate(dist_criteria):
    #             max_tpr = 0.0
    #             min_fdr = 1.0
    #             max_dice = 0.0

    #             for t_idx, threshold in enumerate(thresholds):
    #                 for s_idx, sigma in enumerate(sigmas):
    #                     for h_idx, max_h in enumerate(max_hs):
    #                         # Accumulators and post-processing constants
    #                         avg_tpr, avg_fdr, avg_dice = 0.0, 0.0, 0.0
    #                         for data in self.validation_dataloader:
    #                             imgs, labels, ids = data
    #                             x = self.prepare_data(imgs) # i.e. move to GPU in this case
    #                             y = self(x)

    #                             batch_size = len(imgs)
    #                             for i in range(batch_size):
    #                                 # Perform post processing
    #                                 pred, gt = y[i,0,:,:].detach().cpu(), labels[i,0,:,:].detach().cpu()
    #                                 if distance_transform:
    #                                     proc_loc = postprocess_dt(pred, threshold)
    #                                 else:
    #                                     proc_loc = postprocess(pred, threshold, sigma, max_h)

    #                                 if len(proc_loc.shape) == 2:
    #                                     proc = locations_to_mask(proc_loc, size=(gt.shape[0],gt.shape[1]))
    #                                     if distance_transform:
    #                                         tpr, fdr, dice = evaluate(proc, gt.numpy() == 0, dist_criterium=dist_criterium, hamwoods_free_zone=hamwoods_free_zone)
    #                                     else:
    #                                         tpr, fdr, dice = evaluate(proc, gt.numpy(), dist_criterium=dist_criterium, hamwoods_free_zone=hamwoods_free_zone)
    #                                 else:
    #                                     tpr, fdr, dice = 0.0, 1.0, 0.0

    #                                 avg_tpr += tpr
    #                                 avg_fdr += fdr
    #                                 avg_dice += dice

    #                                 dices[d_idx, t_idx, s_idx, h_idx, ids[i]] = dice
                            
    #                         avg_tpr /= dataset_num
    #                         avg_fdr /=  dataset_num
    #                         avg_dice /=  dataset_num

    #                         print(f"(dist, threshold, sigma, max_h) = ({dist_criterium}, {threshold}, {sigma}, {max_h})")
    #                         print(f"(TPR, FDR, Dice) = ({avg_tpr}, {avg_fdr}, {avg_dice})")

    #                         if avg_tpr > max_tpr:
    #                             max_tpr = avg_tpr
    #                             print(f"Current max TPR!")
                            
    #                         if avg_fdr < min_fdr:
    #                             min_fdr = avg_fdr
    #                             print(f"Current min FDR!")

    #                         if avg_dice > max_dice:
    #                             max_dice = avg_dice
    #                             print(f"Current max Dice!")
    #     self.train()

    #     return max_tpr, min_fdr, max_dice, dices
    
if __name__ == "__main__":
    print("This module should not be executed. Import the FCN from it instead.")