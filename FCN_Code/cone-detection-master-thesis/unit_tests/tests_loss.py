import numpy as np
import torch
import matplotlib.pyplot as plt

from components.loss import MaskedMSELoss
from utils.image import image_to_tensor, tensor_to_image

def test_equivalence():
    dim = 64
    radius = 8
    dx, dy = np.meshgrid(np.arange(dim), np.arange(dim))
    dx, dy = dx - 16, dy - 30
    circle = dx ** 2 + dy ** 2 < radius * radius

    #plt.imshow(circle)
    #plt.show()

    image = image_to_tensor(circle)
    pred = torch.rand((1,1,dim,dim))
    label = torch.rand((1,1,dim,dim))

    loss = MaskedMSELoss()
    better_loss = MaskedMSELoss(better_mask=True)

    phi_loss = loss(image, pred, label)
    phi_better_loss = better_loss(image, pred, label)

    #print(phi_loss.item(), phi_better_loss.item())

    return phi_loss.item() == phi_better_loss.item()
    
def test_difference():
    dim = 64
    radius = 8
    dx, dy = np.meshgrid(np.arange(dim), np.arange(dim))
    dx, dy = dx - 16, dy - 30
    torus_lower = (radius - 3) * (radius - 3) < dx ** 2 + dy ** 2 
    torus_upper = dx ** 2 + dy ** 2 < (radius + 3) * (radius + 3)
    torus = torus_lower & torus_upper

    #plt.imshow(torus)
    #plt.show()

    image = image_to_tensor(torus)
    pred = torch.rand((1,1,dim,dim))
    label = torch.rand((1,1,dim,dim))

    loss = MaskedMSELoss()
    better_loss = MaskedMSELoss(better_mask=True)

    phi_loss = loss(image, pred, label)
    phi_better_loss = better_loss(image, pred, label)

    #print(phi_loss.item(), phi_better_loss.item())

    # Pixels inside the torus should be include with better_loss
    return phi_loss.item() < phi_better_loss.item()