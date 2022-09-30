import torch
import cv2 as cv
import numpy as np
from torchvision import transforms
from PIL import Image
import torch.nn as nn

"""
https://neptune.ai/blog/pytorch-loss-functions

"""

from measurementA import A
from PGGAN import Generator

class ImageAdaptiveGenerator():
    '''
    GAN_type (string): type of pretrained GAN to use
    CSGM_optimizer (string): optimizer used: ADAM, SGD, etc.
    x_path (string): path of image x (original image)
    A_type (string): type of matrix A ('Gaussian', 'Bicubic_Downsample', etc.)
    noise_level (int): noise level 
    scale (float): value of a fraction in the form of 1/x where x > 1
    '''
    def __init__(self, GAN_type, CSGM_optimizer, x_path, A_type, scale=1/8):
        # initialize pre-trained GAN with saved weights in "weights" folder
        if GAN_type == 'PGGAN':
            self.G = Generator()
            self.G.load_state_dict(torch.load('./weights/100_celeb_hq_network-snapshot-010403.pth'))
            self.G.eval() # turn off weights modification in the inference time
        
        # initialize z with normal distribution (0,1) in a form that can be updated by torch
        z_init = torch.normal(mean=0.0, std=1.0, size=(1,512,1,1))
        self.z = torch.autograd.Variable(z_init, requires_grad = True)

        # initialize CSGM optimizer
        self.CSGM_optimizer = CSGM_optimizer

        # initialize A
        if A_type == 'Gaussian':
            self.A = A.guassian_A
        elif A_type == 'Bicubic_Downsample':
            self.A = lambda I: A.bicubic_downsample_A(I, scale)
        else:
            return
        
        # initialize y
        convert_to_tensor = transforms.ToTensor()
        x_PIL = Image.open(x_path)
        self.x = torch.unsqueeze(convert_to_tensor(x_PIL), 0)
        self.y = self.A(self.x)

    def CSGM(self, csgm_iteration_number, csgm_learning_rate):
        # define the cost function
        cost = nn.MSELoss()
        # define the optimizer
        if self.CSGM_optimizer == "SGD":
            optimizer = torch.optim.SGD(params=[self.z], lr=csgm_learning_rate)
        else:
            return

        # CSGM training starts here
        for itr in range(csgm_iteration_number):
            # generate an image from the current z
            Gz = self.G.forward(self.z)
            # create the loss function
            loss = cost(self.y, self.A(Gz))
            # back-propagation
            loss.backward()
            # update z 
            optimizer.step()
            # clear gradient in optimizer
            optimizer.zero_grad()
            # print out each 100th iterations
            if itr % 100 == 0:
                print(f"iteration {itr}, loss = {loss:.10f}")
             
        CSGM_img = self.G(self.z)
        return CSGM_img
    
    def IAGAN(self):
        #optimizes z and theta
        pass 

    def BP(self):
        #enforce compliance 
        pass