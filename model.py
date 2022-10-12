import torch
from torchvision import transforms
from PIL import Image
import torch.nn as nn

"""
https://neptune.ai/blog/pytorch-loss-functions

"""

from A import A
from PGGAN import Generator
from visualizer import savePlot, saveImage, saveTable

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ImageAdaptiveGenerator():
    def __init__(self, GAN_type, CSGM_optimizer, IA_optimizer_z, IA_optimizer_G, x_path, A_type, scale, noise_level, result_folder_name):
        # initialize pre-trained GAN with saved weights in "weights" folder
        if GAN_type == 'PGGAN':
            self.G = Generator().to(device)
            self.G.load_state_dict(torch.load('./weights/100_celeb_hq_network-snapshot-010403.pth', map_location=device))
            self.G.eval() # turn off weights modification in the inference time
        else:
            print('ERROR: pretrained GAN not found')
            exit(0)

        # initialize z with normal distribution (0,1) in a form that can be updated by torch
        z_init = torch.normal(mean=0.0, std=1.0, size=(1,512,1,1)).to(device)
        self.z = torch.autograd.Variable(z_init, requires_grad = True)

        # initialize CSGM optimizer
        self.CSGM_optimizer = CSGM_optimizer

        #initialize IA optimizer 
        self.IA_optimizer_z = IA_optimizer_z
        self.IA_optimizer_G = IA_optimizer_G

        # initialize x
        convert_to_tensor = transforms.ToTensor()
        x_PIL = Image.open(x_path)
        self.x = torch.unsqueeze(convert_to_tensor(x_PIL), 0).to(device)

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
        self.y = self.A(self.x).to(device)
        noise = (torch.rand_like(self.y)*noise_level).to(device)
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
        cost = nn.MSELoss().to(device)
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
            loss = cost(self.y, self.A(Gz).to(device))
            CSGM_loss.append(loss.item())
            # back-propagation
            loss.backward()
            # update z
            optimizer.step()
            # clear gradient in optimizer
            optimizer.zero_grad()
            # print out each 10th iterations
            if (itr+1) % 100 == 0:
                print(f"iteration {itr+1}, loss = {loss:.10f}")
            # save images every 100th image
            if (itr+1) % 600 == 0:
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
        cost = nn.MSELoss().to(device)
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
            loss = cost(self.y, self.A(Gz).to(device))
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
            if (itr+1) % 100 == 0:
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

'''
    Instantiate the model given folder name and image name
'''
def run_model(img, parent_path, img_folder):
    folder_name = parent_path + img_folder

    # Instanciate our IAGAN
    generator = ImageAdaptiveGenerator(
            GAN_type='PGGAN', 
            CSGM_optimizer="ADAM", 
            IA_optimizer_z="ADAM", 
            IA_optimizer_G="ADAM",
            x_path=img,
            A_type="Bicubic_Downsample", 
            noise_level=0/255,
            scale=1/16, 
            result_folder_name=folder_name)
    
    # Orginal "good" image x and degraded image y
    original_x = generator.x
    degraded_y = generator.y
    saveImage(original_x, "original_x", folder_name)
    saveImage(degraded_y, "degraded_y", folder_name)

    # Naive Reconstruction through pseudo-inverse A
    naive_reconstruction = generator.Naive()
    saveImage(naive_reconstruction, "naive_reconstruction", folder_name)

    # Image produced by GAN with the initial z
    GAN_img = generator.GAN()
    saveImage(GAN_img, "GAN_img", folder_name)
    
    # CSGM 
    CSGM_img, CSGM_data = generator.CSGM(csgm_iteration_number=10, csgm_learning_rate=0.1)
    saveImage(CSGM_img, "CSGM_optimized", folder_name)

    # CSGM-BP
    CSGM_BP_img = generator.BP()
    saveImage(CSGM_BP_img, "CSGM_BP", folder_name)

    # IA
    IA_img, IA_data = generator.IA(IA_iteration_number=10, IA_z_learning_rate=0.0001, IA_G_learning_rate=0.001)
    saveImage(IA_img, "IA_optimized", folder_name)

    # IA_BP
    IA_BP_img = generator.BP()
    saveImage(IA_BP_img, "IA_BP", folder_name)

    # Save data as line graphs
    savePlot(CSGM_data, IA_data, folder_name)

    # Save data to a table
    saveTable(original_x, naive_reconstruction, CSGM_img, CSGM_BP_img, IA_img, IA_BP_img, parent_path)
