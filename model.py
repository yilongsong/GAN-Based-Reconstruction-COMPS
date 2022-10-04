from pickletools import optimize
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
from image_saver import saveImage

class ImageAdaptiveGenerator():
    '''
    GAN_type (string): type of pretrained GAN to use
    CSGM_optimizer (string): optimizer used: ADAM, SGD, etc.
    x_path (string): path of image x (original image)
    A_type (string): type of matrix A ('Gaussian', 'Bicubic_Downsample', etc.)
    IA_optimizer_z (string): optimizer used to optimize z
    IA_optimizer_G (string): optimizer used to optimize G
    scale (float): value of a fraction in the form of 1/x where x > 1
    '''
    def __init__(self, GAN_type, CSGM_optimizer, x_path, A_type, IA_optimizer_z, IA_optimizer_G, scale=1/16):
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

        #initialize IA optimizer 
        self.IA_optimizer_z = IA_optimizer_z
        self.IA_optimizer_G = IA_optimizer_G

        # initialize A
        if A_type == 'Gaussian':
            self.A = A.guassian_A
        elif A_type == 'Bicubic_Downsample':
            self.A = lambda I: A.bicubic_downsample_A(I, scale)
            self.A_dag = lambda I: A.bicubic_downsample_A(I, 1/scale)
        else:
            return

        # initialize y
        convert_to_tensor = transforms.ToTensor()
        x_PIL = Image.open(x_path)
        self.x = torch.unsqueeze(convert_to_tensor(x_PIL), 0)
        self.y = self.A(self.x)

    def CSGM(self, csgm_iteration_number, csgm_learning_rate):
        # arrays that store data from CSGM
        CSGM_itr = [i for i in range(csgm_iteration_number)]
        CSGM_loss = []
    
        # define the cost function
        cost = nn.MSELoss()
        # define the optimizer
        if self.CSGM_optimizer == "SGD":
            optimizer = torch.optim.SGD(params=[self.z], lr=csgm_learning_rate)
        elif self.CSGM_optimizer == "ADAM":
            optimizer = torch.optim.Adam(params=[self.z], lr=csgm_learning_rate)

        # CSGM training starts here
        for itr in range(csgm_iteration_number):
            # generate an image from the current z
            Gz = self.G(self.z)
            # save the initial image fron GAN
            if itr == 0:
                original = Gz
            # create the loss function
            loss = cost(self.y, self.A(Gz))
            CSGM_loss.append(loss.item())
            # back-propagation
            loss.backward()
            # update z
            optimizer.step()
            # clear gradient in optimizer
            optimizer.zero_grad()
            # print out each 100th iterations
            if itr % 10 == 0:
                print(f"iteration {itr}, loss = {loss:.10f}")
            # save images
            if itr % 100 == 0:
                saveImage(self.G(self.z), "CSGM_"+str(itr))
        CSGM_img = self.G(self.z)
        return CSGM_img, original, [CSGM_itr, CSGM_loss]

    def IA(self, IA_iteration_number, IA_z_learning_rate, IA_G_learning_rate):
        # arrays that store data from IA
        IA_itr = [i for i in range(IA_iteration_number)]
        IA_loss = []

        # define the cost function
        cost = nn.MSELoss()
        # define the optimizer for z (as of now, ADAM only)
        if self.IA_optimizer_z == "ADAM":
            optimizer_z = torch.optim.Adam(params=[self.z], lr=IA_z_learning_rate)
        # define the optimizer for G (as of now ADAM only)
        if self.IA_optimizer_G == "ADAM":
            optimizer_G = torch.optim.Adam(params=self.G.parameters(), lr=IA_G_learning_rate)

        # unfreeze G's params (maybe)

        # IA steps here
        # CSGM training starts here
        for itr in range(IA_iteration_number):
            # generate an image from the current z
            Gz = self.G(self.z)
            # save the initial image fron GAN
            if itr == 0:
                original = Gz
            # create the loss function
            loss = cost(self.y, self.A(Gz))
            IA_loss.append(loss.item())
            # back-propagation
            loss.backward()
            # update z and G's params
            optimizer_z.step()
            optimizer_G.step()
            # clear gradient in optimizer
            optimizer_z.zero_grad()
            optimizer_G.zero_grad()
            # print out each 100th iterations
            if itr % 10 == 0:
                print(f"iteration {itr}, loss = {loss:.10f}") 
            # save images
            if itr % 100 == 0:
                saveImage(self.G(self.z), "IA_"+str(itr))
        IA_img = self.G(self.z)
        return IA_img, original, [IA_itr, IA_loss]

    def BP(self):
        #enforce compliance
        xhat = self.G(self.z)
        xhat = torch.add(self.A_dag((torch.sub(self.y, self.A(xhat)))), xhat)
        return xhat
