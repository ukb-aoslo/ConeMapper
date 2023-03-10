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

from components.postprocessing import evaluate, _locations_to_mask
from components.loss import DistanceAwareFocalLoss, MaskedMSELoss

from utils.cdc import arcmin_to_pixels, get_circular_mask, get_maximum_extent, pixels_to_arcmin
from utils.image import clip_image

class FCN(nn.Module):
    """
    Fully Convolutional Network as described by 

    'Automatic Detection of Cone Photoreceptors With Fully Convolutional Networks' (Hamwood et al., 2019)
    """

    def __init__(self, train_dataloader, validation_dataloader=None, test_dataloader=None, lr=0.001, depth=3, input_channels=1, initial_feature_maps=16, use_class_weights=True,
                blocks_per_resolution_layer=1, use_MSE_loss=False, masked_MSE_loss=False, use_distance_aware_focal_loss=False):
        super(FCN, self).__init__()

        # Set properties
        self.depth = depth
        self.scaling_factor = 2
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

        # Computed parameters
        self.feature_maps = [input_channels]
        self.feature_maps.extend([self.initial_feature_maps * self.feature_maps_factor ** d for d in range(0,self.depth+1)])
        self.skip_connections_values = []

        # Select device
        use_cuda = torch.cuda.is_available()
        self.device = torch.device("cuda:0" if use_cuda else 'cpu')

        # Parts of the network
        self.encoder = nn.ModuleList()
        self.skip_connections = nn.ModuleList()
        self.center_blocks = nn.ModuleList()
        self.decoder = nn.ModuleList()
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
            return torch.from_numpy(data).float().to(self.device)
        elif type(data) is torch.Tensor:
            return data.float().to(self.device)
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
        self.load_state_dict(torch.load(filepath))

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
                imgs, labels, identifiers = data
                imgs, labels = self.prepare_data(imgs), self.prepare_data(labels) # i.e. move to GPU in this case

                self.optimizer.zero_grad()

                y = self(imgs)

                if self.use_MSE_loss:
                    if self.masked_MSE_loss:
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

            writer.add_scalar('training loss',
                avg_run_loss,
                (current_epoch + 1) * len(self.train_dataloader)) 

            # If a validation DataLoader was given, validate the network
            running_loss = 0.0
            if self.validation_dataloader is not None:
                self.eval()
                with torch.no_grad():
                    for data in self.validation_dataloader:
                        imgs, labels, identifiers = data
                        imgs, labels = self.prepare_data(imgs), self.prepare_data(labels) # i.e. move to GPU in this case

                        y = self(imgs)

                        if self.use_MSE_loss:
                            if self.masked_MSE_loss:
                                loss = self.criterion(imgs, y, labels)
                            else:
                                loss = self.criterion(y, labels)
                        else:
                            loss = self.criterion(y, torch.cat([labels, 1-labels], dim=1))
                        running_loss += loss.item()

                    avg_run_loss = running_loss / len(self.validation_dataloader)

                    if verbose:
                        print("Average running VALIDATION loss for epoch {}: {}".format(current_epoch + 1, avg_run_loss))

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
        if self.validation_dataloader is None:
            self.save("../nets", f"fcn_epoch_{(current_epoch+1):03}_loss_{min_training_loss}")
        writer.close()

        return (current_epoch+1), min_training_loss, min_validation_loss

    @staticmethod
    def train_networks_k_fold_cross_validation(dataloaders, max_epochs=100, early_stopping=False):
        """
        Train instances of a FCN using k-fold cross validation 
        for at most the given number of epochs on each fold.

        Optionally use early stopping.
        """

        for k, (train_dataloader, validation_dataloader) in enumerate(dataloaders):
            model = FCN(train_dataloader, validation_dataloader=validation_dataloader, lr=3e-4, depth=3, input_channels=1, 
                initial_feature_maps=32, blocks_per_resolution_layer=2, use_MSE_loss=True, masked_MSE_loss=True)
            
            trained_epochs, min_training_loss, min_validation_loss = model.train_network(
                max_epochs=max_epochs, 
                run_id=f"fold-{k}", 
                verbose=False, 
                early_stopping=early_stopping)

            print(f"Fold {k}: {trained_epochs} epochs, ({min_training_loss},{min_validation_loss})")
            del model
            torch.cuda.empty_cache()

    # def test_network(self, threshold=0.0, sigma=1.0, max_h=0.05, dist=2.0, free_border=False, save_all_images=False, distance_transform=False,
    #     compute_stats=True, cdc=None, dataset="test"):
    #     """
    #     Test this instance of a FCN with the given
    #     parameters
    #     """

    #     used_dataloader = self.test_dataloader
    #     if dataset == "train":
    #         used_dataloader = self.train_dataloader
    #     if dataset == "validation":
    #         used_dataloader = self.validation_dataloader

    #     if compute_stats:
    #         # Accumulators and post-processing constants
    #         dataset_num = len(used_dataloader.dataset) #len(self.test_dataloader.dataset)
    #         if cdc is not None:
    #             tprs, fdrs, dices, chamfers, gt_cones = np.zeros((dataset_num,30)), np.zeros((dataset_num,30)), np.zeros((dataset_num,30)), np.zeros((dataset_num,30)), np.zeros((dataset_num,30))
    #         else:
    #             tprs, fdrs, dices, chamfers, gt_cones = np.zeros(dataset_num), np.zeros(dataset_num), np.zeros(dataset_num), np.zeros(dataset_num), np.zeros(dataset_num)

    #     self.eval()
    #     with torch.no_grad():
    #         for idx, data in enumerate(used_dataloader): #enumerate(self.test_dataloader):
    #             imgs, labels, ids = data
    #             x = self.prepare_data(imgs) # i.e. move to GPU in this case
    #             y = self(x)

    #             batch_size = len(imgs)
    #             for i in tqdm(range(batch_size)):
    #                 # Perform post processing
    #                 pred, gt = y[i,0,:,:].detach().cpu(), labels[i,0,:,:].detach().cpu()

    #                 if distance_transform:
    #                     processed = postprocess_dt(pred, threshold)
    #                     proc = locations_to_mask(processed, size=(gt.shape[0],gt.shape[1]))

    #                     #if compute_stats:
    #                     #    tpr, fdr, dice = evaluate(proc, gt.numpy() == 0, dist_criterium=dist, hamwoods_free_zone=free_border)
    #                 else:
    #                     proc_loc = postprocess(pred, threshold, sigma, max_h)
    #                     proc = locations_to_mask(proc_loc, size=(gt.shape[0],gt.shape[1]))

    #                     #if compute_stats:
    #                     #    tpr, fdr, dice = evaluate(proc, gt.numpy(), dist_criterium=dist, hamwoods_free_zone=free_border)
                    
    #                 # Regions around CDC
    #                 if cdc is not None:
    #                     cy, cx, _ = cdc[cdc[:,2] == idx][0]
    #                     pixels_per_degree = 600 # 600 is specific to UKB dataset
    #                     extent = get_maximum_extent(gt.shape[0], gt.shape[1], cy, cx, pixels_per_degree) 
    #                     #extent = np.min([extent, 15]) # Cut-off
    #                     #print(extent)
    #                     gt_log = np.zeros(int(extent))
    #                     pred_log = np.zeros(int(extent))
    #                     for eccentricity in range(int(extent)):
    #                         extent_mask = get_circular_mask(gt.shape[0], gt.shape[1], cy, cx, 
    #                             arcmin_to_pixels(pixels_per_degree, eccentricity), arcmin_to_pixels(pixels_per_degree, eccentricity + 1))
    #                         proc_extent = proc * extent_mask
    #                         if distance_transform:
    #                             gt_extent = (gt.numpy() == 0) * extent_mask
    #                         else:
    #                             gt_extent = gt.numpy() * extent_mask
                            
    #                         gt_log[eccentricity] = np.sum(gt_extent)
    #                         pred_log[eccentricity] = np.sum(proc_extent)

    #                         #print(extent, np.sum(proc), np.sum(gt.numpy()), np.sum(proc_extent), np.sum(gt_extent))
    #                         if compute_stats:
    #                             tpr, fdr, dice, chamfer = evaluate(proc_extent, gt_extent, dist_criterium=dist, hamwoods_free_zone=free_border)

    #                             tprs[ids[i],eccentricity] = tpr
    #                             fdrs[ids[i],eccentricity] = fdr
    #                             dices[ids[i],eccentricity] = dice
    #                             chamfers[ids[i],eccentricity] = chamfer
    #                             gt_cones[ids[i],eccentricity] = np.sum(gt_extent)
                        
    #                     xdim = np.arange(int(extent))

    #                     norm = np.pi * (2.0 * xdim + 1)
    #                     gt_log = gt_log / norm * 60 ** 2 # arcmin^2 to deg^2
    #                     pred_log = pred_log / norm * 60 ** 2 #arcmin^2 to deg^2

    #                     _ = plt.figure(figsize=(10,5))
    #                     plt.plot(xdim, gt_log, label="GT")
    #                     plt.plot(xdim, pred_log, label="PRED")
    #                     plt.title(f"Image {ids[i]} - Extent {extent:.2f} [arcmin]")
    #                     plt.ylabel("Radially averaged cone density [cones/deg^2]")
    #                     plt.xlabel("Eccentricity [arcmin]")
    #                     plt.legend()
    #                     plt.savefig(f"figures/temp/cone_density_{ids[i]:06}.png")

    #                     np.savez(f"figures/temp/cone_density_{ids[i]:06}.npz", x=xdim, gt_density=gt_log, pred_density=pred_log)
    #                 # Whole image
    #                 else:
    #                     if compute_stats:
    #                         if distance_transform:
    #                             tpr, fdr, dice, chamfer = evaluate(proc, gt.numpy() == 0, dist_criterium=dist, hamwoods_free_zone=free_border)
    #                         else:
    #                             tpr, fdr, dice, chamfer = evaluate(proc, gt.numpy(), dist_criterium=dist, hamwoods_free_zone=free_border)
                    
    #                 if compute_stats and cdc is None:
    #                     tprs[ids[i]] = tpr
    #                     fdrs[ids[i]] = fdr
    #                     dices[ids[i]] = dice
    #                     chamfers[ids[i]] = chamfer
    #                     gt_cones[ids[i]] = np.sum(gt.numpy())

    #                 def mark_cones(image : np.ndarray, mask : np.ndarray, r : int, g : int, b : int):
    #                     """
    #                     Mark cone locations in an image
    #                     """
    #                     if len(image.shape) == 2:
    #                         color_image = np.stack([image, image, image])
    #                     else:
    #                         color_image = image
    #                     color_image[0,mask == 1] = r
    #                     color_image[1,mask == 1] = g
    #                     color_image[2,mask == 1] = b
    #                     return color_image #.transpose(1,2,0)

    #                 if save_all_images:
    #                     plt.imsave(f"figures/temp/postprocessed_{ids[i]:06}.png", proc)
    #                     plt.imsave(f"figures/temp/image_{ids[i]:06}.png", imgs[i,0,:,:].detach().cpu(), cmap='gray')
    #                     plt.imsave(f"figures/temp/gt_{ids[i]:06}.png", labels[i,0,:,:].detach().cpu())
    #                     plt.imsave(f"figures/temp/prediction_{ids[i]:06}.png", y[i,0,:,:].detach().cpu())

    #                     # Abs diff
    #                     _ = plt.figure(figsize=(5,5), dpi=300)
    #                     plt.imshow(np.abs(labels[i,0,:,:].detach().cpu() - y[i,0,:,:].detach().cpu()))
    #                     plt.title(f"Image {ids[i]} - Absolute difference GT and predicted")

    #                     if cdc is not None:
    #                         cy, cx, _ = cdc[cdc[:,2] == idx][0]
    #                         plt.plot([cx], [cy], "ro", label="CDC")

    #                     plt.legend()
    #                     plt.colorbar()
    #                     plt.savefig(f"figures/temp/abs_diff_pred_gt_{ids[i]:06}.png")

    #                     # Overlay
    #                     _ = plt.figure(figsize=(8,8), dpi=250)

    #                     img = imgs[i,0,:,:].detach().cpu().numpy()
    #                     lab = labels[i,0,:,:].detach().cpu().numpy() == 0 if distance_transform else labels[i,0,:,:].detach().cpu().numpy()

    #                     img = mark_cones(img, lab, 0, 0, 255)
    #                     img = mark_cones(img, proc, 255, 255, 0)

    #                     common = lab * proc

    #                     img = mark_cones(img, common, 0, 255, 0)  

    #                     img = img.transpose(1,2,0)                

    #                     plt.imshow(img)
    #                     plt.title(f"Image {ids[i]} - GT and PR cone locations")

    #                     plt.plot([], [], "bo", label="GT")
    #                     plt.plot([], [], "yo", label="PR")
    #                     plt.plot([], [], "go", label="GT/PR coincide")

    #                     if cdc is not None:
    #                         cy, cx, _ = cdc[cdc[:,2] == idx][0]
    #                         plt.plot([cx], [cy], "mo", label="CDC")

    #                     plt.legend()
    #                     plt.tight_layout()
    #                     plt.savefig(f"figures/temp/overlay_{ids[i]:06}.png")

    #     self.train()
        
    #     if compute_stats:
    #         np.savez("test.npz", tprs=tprs, fdrs=fdrs, dices=dices, chamfers=chamfers, gt_cones=gt_cones)

    #         if cdc is None:
    #             avg_tpr = np.sum(tprs) / dataset_num
    #             avg_fdr = np.sum(fdrs) / dataset_num
    #             avg_dice = np.sum(dices) / dataset_num
    #             avg_chamfer = np.sum(chamfers) / dataset_num

    #             std_tpr = np.sqrt(np.sum((tprs - avg_tpr) ** 2) / dataset_num)
    #             std_fdr = np.sqrt(np.sum((fdrs - avg_fdr) ** 2) / dataset_num)
    #             std_dice = np.sqrt(np.sum((dices - avg_dice) ** 2) / dataset_num)
    #             std_chamfer = np.sqrt(np.sum((chamfers - avg_chamfer) ** 2) / dataset_num)

    #             print(f"Input test images: {dataset_num}")
    #             print(f"Average true positive rate: {avg_tpr} (STD: {std_tpr})")
    #             print(f"Average false detection rate: {avg_fdr} (STD: {std_fdr})")
    #             print(f"Average dice coefficient: {avg_dice} (STD: {std_dice})")
    #             print(f"Average Chamfer distance: {avg_chamfer} (STD: {std_chamfer})")

    #             with open("figures/temp/stats.txt", "w") as f:
    #                 f.write(f"{dataset_num} input images\n")
    #                 if distance_transform:
    #                     f.write(f"Post-processing with threshold={threshold} distance={dist}\n")
    #                 else:
    #                     f.write(f"Post-processing with sigma={sigma} threshold={threshold} max_h={max_h} distance={dist}\n")
    #                 f.write(f"Average true positive rate: {avg_tpr} (STD: {std_tpr})\n")
    #                 f.write(f"Average false detection rate: {avg_fdr} (STD: {std_fdr})\n")
    #                 f.write(f"Average dice coefficient: {avg_dice} (STD: {std_dice})\n")
    #                 f.write(f"Average Chamfer distance: {avg_chamfer} (STD: {std_chamfer})\n")

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