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
        elif self.CSGM_optimizer == "ADAM":
            optimizer = torch.optim.Adam(params=[self.z], lr=csgm_learning_rate)

        #original = self.z.detach().clone()
        #print(original)
        original = None

        # CSGM training starts here
        for itr in range(csgm_iteration_number):
            # generate an image from the current z
            Gz = self.G(self.z)
            if itr == 0:
                original = Gz
            # create the loss function
            loss = cost(self.y, self.A(Gz))
            # back-propagation
            loss.backward()
            # update z
            optimizer.step()
            # clear gradient in optimizer
            optimizer.zero_grad()
            # print out each 100th iterations
            if itr % 10 == 0:
                print(f"iteration {itr}, loss = {loss:.10f}")
                #print(torch.sum(torch.abs(self.z - original)))
        CSGM_img = self.G(self.z)
        return CSGM_img, original

    # def IAGAN(self):
    #     """
    #     self.G.features = self.G.features.(requires_grad = True)
    #     self.G.output = self.G.output.(requires_grad = True)
    #     CSGM_img = self.CSGM(self, csgm_iteration_number, csgm_learning_rate)
    #     return CSGM_img
    #     """
    #     return

    # def BP(self):
    #     #enforce compliance
    #     xhat = self.G(self.z)
    #     A_dag = torch.matmul(torch.t(A), torch.inverse(torch.matmul(A,torch.t(A))))
    #     xhat = torch.add(torch.matmul(A_dag,(torch.sub(self.y, torch.matmul(A, xhat)))), xhat)
    #     return xhat
