import torch.nn as nn
import torch
import torch.optim as optim
from .configuration import lr, beta1, device, nz

def optimizer(module):
    return optim.Adam(module.parameters(), lr=lr, betas=(beta1, 0.999))


# Initialize BCELoss function
criterion = nn.BCELoss()

# Create batch of latent vectors that we will use to visualize
#  the progression of the generator
fixed_noise = torch.randn(64, nz, 1, 1, device=device)
