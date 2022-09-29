import torch
import cv2 as cv
import numpy as np
from torchvision import transforms
from PIL import Image

from PGGAN import Generator

G = Generator()
G.load_state_dict(torch.load('./weights/100_celeb_hq_network-snapshot-010403.pth'))

z_init = torch.normal(torch.zeros((5,512,1,1)))
z = torch.autograd.Variable(z_init[0:1, :], requires_grad = True)

Gz = G(z)

convert_to_PIL = transforms.ToPILImage()
A_image = convert_to_PIL((Gz[0]+1)/2)
A_image.show()