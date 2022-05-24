import torch
import torch.nn as nn
import random

from functions.train import train
from functions.configuration import ngpu, num_epochs
from functions.preprocess import dataloader, device, weights_init
from functions.generator import Generator
from functions.discriminator import Discriminator
from functions.result import plot
from functions.visualize import visualize

# Set random seed for reproducibility
manualSeed = 999
#manualSeed = random.randint(1, 10000) # use if you want new results
print("Random Seed: ", manualSeed)
random.seed(manualSeed)
torch.manual_seed(manualSeed)

# Generator
netG = Generator().to(device)
# Discriminator
netD = Discriminator().to(device)

if device.type == "cuda" and ngpu > 1:
    netG = nn.parallel.DataParallel(netG, list(range(ngpu)))
    netD = nn.parallel.DataParallel(netD, list(range(ngpu)))

netG.apply(weights_init)
netD.apply(weights_init)

print(f"netG: {netG}")
print(f"netD: {netD}")

img_list, D_losses, G_losses = train(num_epochs=num_epochs, dataloader=dataloader, netD=netD, netG=netG)
plot(D_losses, G_losses)

visualize(img_list)