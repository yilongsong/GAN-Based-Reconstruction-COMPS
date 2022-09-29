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

    def guassian_A():
        '''
        mu, sigma = 0, 1/np.sqrt(1024)
        A = np.random.normal(mu, sigma, (1024, 1024, 3))
        apply_A = lambda img : np.multiply(A, img)
        '''
        pass

    def bicubic_downsample_A():
        pass