import torch
import torch.nn as nn
import torch.nn.functional as func
import torchvision
import numpy as np

class DistanceAwareFocalLoss(nn.Module):
    """
    Distance Aware Focal Loss
    (see CornerNet: Detecting Objects as Paired Keypoints, Law et al., ECCV 2018)
    """
    def __init__(self, alpha=1.0, beta=1.0):
        super(DistanceAwareFocalLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta

        # A deviation of 2-3 pixels in each direction should be tolerated
        kernel_size = (5,5)
        sigma = 0.5 #1.1 #0.4 #0.5
        self.weighting_function = torchvision.transforms.GaussianBlur(kernel_size, sigma=(sigma,sigma))
        #self.weighting_scaling = sigma * np.sqrt(2.0 * np.pi) # normalize center to 1.0
        self.weighting_scaling = sigma ** 2 * 2.0 * np.pi # normalize center to 1.0

    # labels = (B,2,H,W)
    def forward(self, input, labels):
        a = torch.log(input + 1e-7) # Input should not be exactly 0.0
        b = torch.log(1.0 + 1e-7 - input) # Input should not be exactly 1.0

        # Use Gaussian weighting
        #weighting = labels
        #for i in range(labels.shape[0]):
        #    weighting[i,:,:,:] = self.weighting_scaling * self.weighting_function(weighting[i,:,:,:])
        weighting = self.weighting_scaling * self.weighting_function(labels)

        #c = labels * (1.0 - input) ** self.alpha * a
        #d = (1.0 - labels) * (1.0 - weighting) ** self.beta * input ** self.alpha * b
        c = (weighting >= 1.0) * (1.0 - input) ** self.alpha * a
        d = (weighting < 1.0) * (1.0 - weighting) ** self.beta * input ** self.alpha * b

        return -1.0 * (c + d).mean()

class MaskedMSELoss(nn.Module):
    """
    Masked MSE Loss
    (= masking black background to not contribute)
    """
    def __init__(self, threshold=1.0/255.0, better_mask=False):
        """
        threshold = cut-off for images in range [0,1]
        """
        super(MaskedMSELoss, self).__init__()
        self.threshold = threshold
        self.better_mask = better_mask

    # labels = (B,1,H,W)
    def forward(self, image, prediction, label):
        if self.better_mask:
            # Compute mask by flooding from all 4 sides
            top = torch.cumsum(image, 2) < self.threshold
            bottom = torch.flip(torch.cumsum(torch.flip(image, [2]), 2), [2]) < self.threshold
            left = torch.cumsum(image, 3) < self.threshold
            right = torch.flip(torch.cumsum(torch.flip(image, [3]), 3), [3]) < self.threshold
            # Only include pixels where all directions are False (= not part of the black border)
            mask = ~(top | bottom | left | right) 
        else:
            # Just exclude all black pixels
            mask = (image >= self.threshold).float()  
        
        # Compute and return loss 
        prediction = prediction * mask
        label = label * mask
        loss = torch.mean(torch.pow(torch.subtract(prediction, label), 2.0))
        return loss