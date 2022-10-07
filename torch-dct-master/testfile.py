import torch
import torch_dct as dct
import cv2 
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
import scipy

def create_mask(ratio, row_size, col_size):
    mask = torch.zeros(row_size, col_size)
    target = int(ratio*row_size*col_size)
    while (target > 0):
        x = int(np.mod(np.floor(np.abs(np.random.randn())*row_size), row_size))
        y = int(np.mod(np.floor(np.abs(np.random.randn())*col_size), col_size))
        if mask[x][y] != 1:
            mask[x,y] = 1
            target -= 1
    return mask