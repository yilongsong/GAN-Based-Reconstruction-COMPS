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

def main():
    path = "../Images/CelebA_HQ/000168.jpg"
    img = Image.open(path)
    convert_to_PIL = transforms.ToPILImage()
    convert_to_tensor = transforms.ToTensor()

    x = convert_to_tensor(img) # good image

    #X = scipy.fftpack.dct(x, type=3, n=10, axis=-1, norm=None, overwrite_x=False)

    #X = dct.dct_3d(x)   # DCT-II done through the last dimension, shape = [3,1024,1024]
    #y = dct.idct_3d(X)  # scaled DCT-III done through the last dimension

    X = torch.fft.fftn(x, s=None, dim=3, norm=None, out=None)
    y = torch.fft.ifftn(X, s=None, dim=3, norm=None, out=None)


    X = Image.fromarray((255*X[0]).numpy().astype(np.uint8).transpose(1, 2, 0))
    #X = convert_to_PIL(X)
    #y = convert_to_PIL(Y)
    Y = Image.fromarray((255*Y[0]).numpy().astype(np.uint8).transpose(1, 2, 0))

    X.show()
    y.show()

def create_mask(size=1024, r=1024):
    mask = torch.zeros(size, size)
    x = np.mod(np.floor(np.abs(np.random.randn(size, size))*r), size)
    y = np.mod(np.floor(np.abs(np.random.randn(size, size))*r), size)
    mask[x,y] = 1
    ratio = np.sum(mask.numpy()==1)/mask.numpy().size
    return mask, ratio

mask, ratio = create_mask()
print(ratio)

def compression(x, mask):
    x_transformed = torch.empty((3,1024,1024))
    for i in range(3):
        ch = x[i]
        x_transformed[i] = torch.fft.fft2(ch)
    
    x_masked = torch.zeros((3,1024,1024))
    for ch in range(3):
        for i in range(1024):
            for j in range(1024):
                if mask[i][j] == 1:
                    x_masked[ch][i][j] = x_transformed[ch][i][j]

    
     
compression(x,mask)
