'''
    Where we implement measurement metrecies A

    https://discuss.pytorch.org/t/pytorch-pil-to-tensor-and-vice-versa/6312
'''
import torch
import cv2 
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

from torchvision import transforms

class A():
    def __init__(self):
        pass

    def guassian_A(img) -> torch.tensor:
        A = torch.normal(mean=0.0, std=1/np.sqrt(img.shape[2]), size=(1,3,1024,1024)) 
        return torch.multiply(A, img)

    def bicubic_downsample_A(img, scale) -> torch.tensor:
        return torch.nn.functional.interpolate(img, scale_factor=scale, mode='bicubic', align_corners=False, antialias=True) 