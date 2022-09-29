import torch
import cv2 as cv
import numpy as np
from torchvision import transforms
from PIL import Image

from measurementA import A
from PGGAN import Generator

G = Generator()
G.load_state_dict(torch.load('./weights/100_celeb_hq_network-snapshot-010403.pth'))

z_init = torch.normal(mean=0.0, std=1.0, size=(1,512,1,1))
z = torch.autograd.Variable(z_init, requires_grad = True) # this step is required in the future because 
# we need gradients to be computed for z_init

Gz = G(z)
AGz1 = A.guassian_A(Gz)
AGz2 = A.bicubic_downsample_A(Gz, 0.0625)

convert_to_PIL = transforms.ToPILImage()

# Generated Image
A_image = convert_to_PIL((Gz[0]+1)/2)
A_image.show()

# Comprssed 
AGz1_image = convert_to_PIL(AGz1[0]/2)
AGz1_image.show()

# Low Resolution
AGz2_image = convert_to_PIL((AGz2[0]+1)/2)
AGz2_image.show()