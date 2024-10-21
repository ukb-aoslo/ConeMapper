import torch
import torch.nn as nn
import torch.nn.functional as func
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import os
from datetime import datetime
from tqdm import tqdm

class Hamwood(nn.Module):
    """
    Python port of Hamwood et al. (2019) from MATLAB
    """
    def get_block(self, channels):
        return nn.ModuleList([
                nn.LazyConv2d(channels, (3,3), padding=(1,1)),
                nn.BatchNorm2d(channels),
                nn.ReLU(),
                nn.LazyConv2d(channels, (3,3), padding=(1,1)),
                nn.BatchNorm2d(channels),
                nn.ReLU()
            ])

    def __init__(self, train_dataloader, validation_dataloader=None, batch_size=32):
        super(Hamwood, self).__init__()

        # Props
        self.lr = 0.001
        self.train_dataloader = train_dataloader
        self.validation_dataloader = validation_dataloader
        self.batch_size = batch_size

        # Select device
        use_cuda = torch.cuda.is_available()
        self.device = torch.device("cuda:0" if use_cuda else 'cpu')

        # Build architecture
        self.encoder1 = self.get_block(32) 
        self.maxPool1 = nn.MaxPool2d((2,2), (2,2))

        self.encoder2 = self.get_block(32) 
        self.maxPool2 = nn.MaxPool2d((2,2), (2,2))

        self.encoder3 = self.get_block(32)
        self.maxPool3 = nn.MaxPool2d((2,2), (2,2))

        self.encoder4 = self.get_block(256)

        self.dropout = nn.Dropout2d(p=0.5)

        self.transConv4 = nn.LazyConvTranspose2d(128, (4,4), stride=(2,2), padding=(1,1))
        self.decoder4 = self.get_block(128)

        self.transConv3 = nn.LazyConvTranspose2d(64, (4,4), stride=(2,2), padding=(1,1))
        self.decoder3 = self.get_block(128)

        self.transConv2 = nn.LazyConvTranspose2d(32, (4,4), stride=(2,2), padding=(1,1))
        self.decoder2 = self.get_block(128)
            
        self.conv = nn.LazyConv2d(2, (1,1))
        self.softmax = nn.Softmax2d()

        # Move model to preferred device
        self.to(self.device)

        # Criterion and Optimizer      
        weights = torch.Tensor([1.0 - 0.02, 1.0 - 0.98]).to(self.device)     
        self.criterion = nn.CrossEntropyLoss(weight=weights)
        self.optimizer = optim.SGD(self.parameters(), lr=self.lr)

        # Batch-Size: 20
        # Max-Epochs: 30
        # Execution-Environment: gpu
        # Shuffle: every-epoch (WTF?)


    def forward(self, x):
        enc1 = x
        for module in self.encoder1:
            enc1 = module(enc1)
        x = self.maxPool1(enc1)

        enc2 = x
        for module in self.encoder2:
            enc2 = module(enc2)
        x = self.maxPool2(enc2)

        enc3 = x
        for module in self.encoder3:
            enc3 = module(enc3)
        x = self.maxPool3(enc3)

        enc4 = x
        for module in self.encoder4:
            enc4 = module(enc4)

        x = self.dropout(enc4)

        x = self.transConv4(x)
        x = torch.cat([x, enc3], dim=1)
        for module in self.decoder4:
            x = module(x)

        x = self.transConv3(x)
        x = torch.cat([x, enc2], dim=1)
        for module in self.decoder3:
            x = module(x)

        x = self.transConv2(x)
        x = torch.cat([x, enc1], dim=1)
        for module in self.decoder2:
            x = module(x)

        x = self.conv(x)
        x = self.softmax(x)

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
                imgs, labels, identifiers, cdc, distances = data

                self.optimizer.zero_grad()

                imgs, labels = imgs.to(self.device).float(), labels.to(self.device).float()
                y = self(imgs)

                classes = torch.cat([labels, 1-labels], dim=1)
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
                        imgs, labels, identifiers, cdc, distances = data                     

                        imgs, labels = imgs.to(self.device).float(), labels.to(self.device).float()
                        y = self(imgs)

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
                            self.save("../nets", f"hamwood_epoch_{(current_epoch+1):03}_loss_{min_validation_loss}")
                    elif early_stopping and avg_run_loss > 1.2 * min_validation_loss:
                        # Stop if the validation loss does not improve
                        return (current_epoch+1), min_training_loss, min_validation_loss

                    writer.add_scalar('validation loss',
                        avg_run_loss,
                        (current_epoch + 1) * len(self.validation_dataloader))
                self.train()

        # Clean up after training
        if self.validation_dataloader is None:
            self.save("../nets", f"hamwood_epoch_{(current_epoch+1):03}_loss_{min_training_loss}")
        writer.close()

        return (current_epoch+1), min_training_loss, min_validation_loss

    @staticmethod
    def train_network_single_cross_validation(fold_id, train_dataloader, validation_dataloader, max_epochs=100, early_stopping=False):
        """
        Train an instance of Hamwood using cross validation 
        for at most the given number of epochs with a single fold.

        Optionally use early stopping.
        """
        model = Hamwood(train_dataloader, validation_dataloader=validation_dataloader, batch_size=32)
            
        trained_epochs, min_training_loss, min_validation_loss = model.train_network(
            max_epochs=max_epochs, 
            run_id=f"fold-{fold_id}", 
            verbose=False, 
            early_stopping=early_stopping)

        print(f"Fold {fold_id}: {trained_epochs} epochs, ({min_training_loss},{min_validation_loss})")
        del model
        torch.cuda.empty_cache()

    @staticmethod
    def train_networks_k_fold_cross_validation(dataloaders, max_epochs=100, early_stopping=False):
        """
        Train instances of Hamwood using k-fold cross validation 
        for at most the given number of epochs on each fold.

        Optionally use early stopping.
        """

        for k, (train_dataloader, validation_dataloader) in enumerate(dataloaders):
            model = Hamwood(train_dataloader, validation_dataloader=validation_dataloader, batch_size=32)
            
            trained_epochs, min_training_loss, min_validation_loss = model.train_network(
                max_epochs=max_epochs, 
                run_id=f"fold-{k}", 
                verbose=False, 
                early_stopping=early_stopping)

            print(f"Fold {k}: {trained_epochs} epochs, ({min_training_loss},{min_validation_loss})")
            del model
            torch.cuda.empty_cache()