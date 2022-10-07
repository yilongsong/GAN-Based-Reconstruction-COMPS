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

    X = torch.fft.fft(x)
    Y = torch.fft.ifft(X)


    X = Image.fromarray((255*X[0]).numpy().astype(np.uint8).transpose(1, 2, 0))
    #X = convert_to_PIL(X)
    #y = convert_to_PIL(Y)
    Y = Image.fromarray((255Y[0]).numpy().astype(np.uint8).transpose(1, 2, 0))

    X.show()
    y.show()

if __name__ == '__main__':
    main()