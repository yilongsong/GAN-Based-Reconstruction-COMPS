'''
    Where we implement measurement metricies A

    https://discuss.pytorch.org/t/pytorch-pil-to-tensor-and-vice-versa/6312
'''
import torch
import cv2 
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from itertools import product
from torchvision import transforms
import DCT.torch_dct as dct

convert_to_PIL = transforms.ToPILImage()
convert_to_tensor = transforms.ToTensor()

class A():
    def __init__(self):
        pass
        
    def bicubic_downsample_A(img, scale) -> torch.tensor:
        return torch.nn.functional.interpolate(img, scale_factor=scale, mode='bicubic', align_corners=False, antialias=True)

    def create_simple_mask(img, ratio):
        rows = img.shape[2]
        cols = img.shape[3]
        mask = torch.zeros((rows, cols))
        a = torch.tensor(list(product(range(rows), range(cols))))
        prob = torch.tensor([1/(rows*cols)]*rows*cols)
        idx = prob.multinomial(num_samples=int(rows*cols*ratio), replacement=False)
        for i in a[idx]:
            mask[i[0], i[1]] = 1
        return mask

    def simple_compression_A(mask, img):
        return torch.mul(mask, img)

    def dct_compression_A(img, ratio=None):
        return dct.dct_2d(img)   # DCT-II done through the last dimension

    def idct_compression_A(img, ratio=None):
        return dct.idct_2d(img)  # scaled DCT-III done through the last dimension