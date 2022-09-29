'''
    Where we implement measurement metrecies A
'''
import torch
import cv2 
import numpy as np
import matplotlib.pyplot as plt

class A():
    def __init__(self, matrix_type):
        # depending on type
        # assign self.A = lambda image: "operation"
        pass

    def guassian_A(img) -> torch.tensor:
        A = torch.normal(mean=0.0, std=1/np.sqrt(img.shape[2]), size=(1,3,1024,1024)) 
        return torch.multiply(A, img)

    def bicubic_downsample_A(img, scale) -> torch.tensor:
        return torch.nn.functional.interpolate(img, scale_factor=scale, mode='bicubic')
        