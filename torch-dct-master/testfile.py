import torch
import torch_dct as dct
import cv2 
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms

import scipy

x = torch.randn(200)
x = x[None, :]
x = x[None, :]
X = dct.dct_3d(x)   # DCT-II done through the last dimension
y = dct.idct_3d(X)  # scaled DCT-III done through the last dimension
#print((torch.abs(x - y)).sum())
#print((torch.abs(x - y)).sum() < 1e-10)
#assert (torch.abs(x - y)).sum() < 1e-10  # x == y within numerical tolerance

path = "../Images/CelebA_HQ/000168.jpg"
img = Image.open(path)
convert_to_PIL = transforms.ToPILImage()
convert_to_tensor = transforms.ToTensor()

x = convert_to_tensor(img) # good image

def compression_test():
    path = "../Images/CelebA_HQ/000168.jpg"
    img = Image.open(path)
    convert_to_PIL = transforms.ToPILImage()
    convert_to_tensor = transforms.ToTensor()

    x = convert_to_tensor(img) # good image

    #X = scipy.fftpack.dct(x, type=3, n=10, axis=-1, norm=None, overwrite_x=False)

    #X = dct.dct_3d(x)   # DCT-II done through the last dimension, shape = [3,1024,1024]
    #y = dct.idct_3d(X)  # scaled DCT-III done through the last dimension

    X = torch.fft.fftn(x, s=None, dim=3, norm=None, out=None)
    Y = torch.fft.ifftn(X, s=None, dim=3, norm=None, out=None)

    X = convert_to_PIL(X)
    y = convert_to_PIL(y)

    X.show()
    y.show()

def create_mask(size=1024, ):
    pass

def compression():
    pass