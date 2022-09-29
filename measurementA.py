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

    def guassian_A(Gz) -> torch.tensor:
        A = torch.normal(mean=0.0, std=1/np.sqrt(Gz.shape[2]), size=(1,3,1024,1024)) 
        return torch.multiply(A, Gz)

    def bicubic_downsample_A(Gz, scale) -> torch.tensor:
        return torch.nn.functional.interpolate(Gz, scale_factor=scale, mode='bicubic')
        