import torch
from torchvision import transforms
from PIL import Image
import torch.nn as nn

"""
https://neptune.ai/blog/pytorch-loss-functions

"""

from A import A
from PGGAN import Generator
from visualizer import saveImage

class ImageAdaptiveGenerator():
    def __init__(self, GAN_type, CSGM_optimizer, IA_optimizer_z, IA_optimizer_G, x_path, A_type, scale, noise_level, result_folder_name):
        # initialize pre-trained GAN with saved weights in "weights" folder
        if GAN_type == 'PGGAN':
            self.G = Generator()
            self.G.load_state_dict(torch.load('./weights/100_celeb_hq_network-snapshot-010403.pth'))
            self.G.eval() # turn off weights modification in the inference time
        else:
            print('ERROR: pretrained GAN not found')
            exit(0)

        # initialize z with normal distribution (0,1) in a form that can be updated by torch
        z_init = torch.normal(mean=0.0, std=1.0, size=(1,512,1,1))
        self.z = torch.autograd.Variable(z_init, requires_grad = True)

        # initialize CSGM optimizer
        self.CSGM_optimizer = CSGM_optimizer

        #initialize IA optimizer 
        self.IA_optimizer_z = IA_optimizer_z
        self.IA_optimizer_G = IA_optimizer_G

        # initialize x
        convert_to_tensor = transforms.ToTensor()
        x_PIL = Image.open(x_path)
        self.x = torch.unsqueeze(convert_to_tensor(x_PIL), 0)

        # initialize A
        if A_type == 'Naive_Compression':
            mask = A.create_simple_mask(self.x, scale)
            self.A = lambda I: A.simple_compression_A(I, mask)
            self.A_dag = self.A
        elif A_type == 'DCT_Compression':
            self.A = lambda I: A.dct_compression_A(I)
            self.A_dag = lambda I: A.idct_compression_A(I)
        elif A_type == 'Bicubic_Downsample':
            self.A = lambda I: A.bicubic_downsample_A(I, scale)
            self.A_dag = lambda I: A.bicubic_downsample_A(I, 1/scale)
        else:
            print('ERROR: A not found')
            exit(0)

        # initialize y with given noise_level
        self.y = self.A(self.x)
        noise = torch.rand_like(self.y)*noise_level
        self.y += noise

        # folder that all images will be stored
        self.result_folder_name = result_folder_name

    '''
        Return a naive reconstruction of y obtained through A_dag
        @params: None
        @return: a naive reconstruction of y
    '''
    def Naive(self):
        return self.A_dag(self.y)

    '''
        Return an image produced by GAN with the initial z
        @params: None
        @return: G(z)
    '''
    def GAN(self):
        return self.G(self.z)

    '''
        Perform CSGM given the number of iteration and learning rate
        @params: csgm_iteration_number - # of iterations
                 csgm_learning_rate - the learning rate
        @return: CSGM_img - the image obtained by CSGM
                 [CSGM_itr, CSGM_loss] - a list containing data of change in loss function
    '''
    def CSGM(self, csgm_iteration_number, csgm_learning_rate):
        print("Launching CSGM optimization:")
    
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
        else:
            print('ERROR: CSGM optimizer not found')
            exit(0)

        # CSGM training starts here
        for itr in range(csgm_iteration_number):
            # generate an image from the current z
            Gz = self.G(self.z)
            # create the loss function
            loss = cost(self.y, self.A(Gz))
            CSGM_loss.append(loss.item())
            # back-propagation
            loss.backward()
            # update z
            optimizer.step()
            # clear gradient in optimizer
            optimizer.zero_grad()
            # print out each 10th iterations
            if (itr+1) % 10 == 0:
                print(f"iteration {itr+1}, loss = {loss:.10f}")
            # save images every 100th image
            if (itr+1) % 100 == 0:
                saveImage(self.G(self.z), "CSGM_"+str(itr+1), self.result_folder_name)

        CSGM_img = self.G(self.z)
        print("CSGM completed")
        return CSGM_img, [CSGM_itr, CSGM_loss]

    '''
        Perform IA given the number of iteration and learning rates
        @params: IA_iteration_number - # of iterations
                 IA_z_learning_rate - the learning rate for z
                 IA_G_learning_rate - the learning rate for G
        @return: IA_img - the image obtained by CSGM
                 [IA_itr, IA_loss] - a list containing data of change in loss function
    '''
    def IA(self, IA_iteration_number, IA_z_learning_rate, IA_G_learning_rate):
        print("Launching IA optimization:")
        # arrays that store data from IA
        IA_itr = [i for i in range(IA_iteration_number)]
        IA_loss = []

        # define the cost function
        cost = nn.MSELoss()
        # define the optimizer for z (as of now, ADAM only)
        if self.IA_optimizer_z == "ADAM":
            optimizer_z = torch.optim.Adam(params=[self.z], lr=IA_z_learning_rate)
        else:
            print('ERROR: CSGM optimizer for z not found')
            exit(0)
        # define the optimizer for G (as of now ADAM only)
        if self.IA_optimizer_G == "ADAM":
            optimizer_G = torch.optim.Adam(params=self.G.parameters(), lr=IA_G_learning_rate)
        else:
            print('ERROR: IA optimizer for G not found')
            exit(0)

        # IA steps here
        for itr in range(IA_iteration_number):
            # generate an image from the current z
            Gz = self.G(self.z)
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
            # print out each 10th iterations
            if (itr+1) % 10 == 0:
                print(f"iteration {itr+1}, loss = {loss:.10f}") 
            # save images every 100th image
            if (itr+1) % 100 == 0:
                saveImage(self.G(self.z), "IA_"+str(itr+1), self.result_folder_name)

        IA_img = self.G(self.z)
        print("IA completed")
        return IA_img, [IA_itr, IA_loss]

    '''
        Perform BP through A_dag
        @params: None
        @return: x_hat - the image obtained by BP
    '''
    def BP(self):
        #enforce compliance
        xhat = self.G(self.z)
        xhat = torch.add(self.A_dag((torch.sub(self.y, self.A(xhat)))), xhat)
        return xhat
