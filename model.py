import torch
import torch.nn as nn
from re import search
from torchvision import transforms
from PIL import Image

"""
https://neptune.ai/blog/pytorch-loss-functions

"""

from A import A
from PGGAN import Generator
from visualizer import saveImage, saveTable

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ImageAdaptiveGenerator():
    def __init__(self, GAN_type, x_path, task, scale, noise_level, result_folder_name):
        # initialize pre-trained GAN with saved weights in "weights" folder
        if GAN_type == 'PGGAN':
            self.G = Generator().to(device)
            self.G.load_state_dict(torch.load('./weights/PGGAN_weights.pth', map_location=device))
            self.G.eval() # turn off weights modification in the inference time

        # initialize z with normal distribution (0,1) in a form that can be updated by torch
        z_init = torch.normal(mean=0.0, std=1.0, size=(1,512,1,1)).to(device)
        self.z = torch.autograd.Variable(z_init, requires_grad = True)

        # initialize x
        convert_to_tensor = transforms.ToTensor()
        x_PIL = Image.open(x_path)
        self.x = torch.unsqueeze(convert_to_tensor(x_PIL), 0).to(device)

        # initialize A
        if task == 'FFT':
            mask = A.render_mask(self.x, scale).to(device)
            self.A = lambda I: A.fft_compression_A(I, mask)
            self.A_dag = lambda I: A.ifft_compression_A(I, mask)
        elif task == 'Bicubic':
            self.A = lambda I: A.bicubic_downsample_A(I, scale)
            self.A_dag = lambda I: A.bicubic_downsample_A(I, 1/scale)
        elif task == 'Blur':
            self.A = lambda I: A.blur_A(I, scale)
            self.A_dag = None

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
    
        # define the cost function
        cost = nn.MSELoss()
        # define the optimizer
        optimizer = torch.optim.Adam(params=[self.z], lr=csgm_learning_rate)

        # CSGM training starts here
        for itr in range(csgm_iteration_number):
            # clear gradient in optimizer
            optimizer.zero_grad()
            # generate an image from the current z
            Gz = self.G(self.z)
            # create the loss function
            loss = cost(self.y, self.A(Gz))
            # back-propagation
            loss.backward()
            # update z
            optimizer.step()
            # print out each 10th iterations
            if (itr+1) % 100 == 0:
                print(f"iteration {itr+1}, loss = {loss:.10f}")

        CSGM_img = self.G(self.z)
        print("CSGM completed")
        return CSGM_img

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

        # define the cost function
        cost = nn.MSELoss()
        # define the optimizer for z (as of now, ADAM only)
        optimizer_z = torch.optim.Adam(params=[self.z], lr=IA_z_learning_rate)
        # define the optimizer for G (as of now ADAM only)
        optimizer_G = torch.optim.Adam(params=self.G.parameters(), lr=IA_G_learning_rate)

        # IA steps here
        for itr in range(IA_iteration_number):
            # clear gradient in optimizer
            optimizer_z.zero_grad()
            optimizer_G.zero_grad()
            # generate an image from the current z
            Gz = self.G(self.z)
            # create the loss function
            loss = cost(self.y, self.A(Gz))
            # back-propagation
            loss.backward()
            # update z and G's params
            optimizer_z.step()
            optimizer_G.step()
            # print out each 100th iterations
            if (itr+1) % 100 == 0:
                print(f"iteration {itr+1}, loss = {loss:.10f}") 

        IA_img = self.G(self.z)
        print("IA completed")
        return IA_img

    '''
        Perform BP through A_dag
        @params: None
        @return: x_hat - the image obtained by BP
    '''
    def BP(self):
        #enforce compliance
        print("Launching BP:")
        xhat = self.G(self.z)
        xhat = torch.add(self.A_dag((torch.sub(self.y, self.A(xhat)))), xhat)
        print("BP complete")
        return xhat

'''
    Instantiate the model given folder name and image name
'''
def run_model(img, params):
    # create a folder that stores our result
    img_path = params['img_dir']+'/'+img
    folder_name = params['parent_path'] + search(r'\d+', img_path).group()

    # Instanciate our IAGAN
    generator = ImageAdaptiveGenerator(
        GAN_type=params['GAN'], 
        x_path=img_path,
        task=params['task'], 
        noise_level=params['noise_level'],
        scale=params['rate'], 
        result_folder_name=folder_name)
    
    # Orginal "good" image x and degraded image y
    original_x = generator.x
    degraded_y = generator.y

    # Naive Reconstruction through pseudo-inverse A
    if params['task'] != 'Blur':
        naive_reconstruction = generator.Naive()

    # Image produced by GAN with the initial z
    GAN_img = generator.GAN()
    
    if params['skip_csgm'] == False:
        # CSGM 
        CSGM_img = generator.CSGM(csgm_iteration_number=params['CSGM_itr'], csgm_learning_rate=params['CSGM_lr'])

        # CSGM-BP
        if params['task'] != 'Blur':
            CSGM_BP_img = generator.BP()

    # IA
    IA_img = generator.IA(IA_iteration_number=params['IA_itr'], IA_z_learning_rate=params['IA_z_lr'], IA_G_learning_rate=params['IA_G_lr'])

    # IA_BP
    if params['task'] != 'Blur':
        IA_BP_img = generator.BP()

    # Save images
    if params['save_images']:
        saveImage(original_x, "original_x", folder_name)
        if params['task'] != 'FFT':
            saveImage(degraded_y, "degraded_y", folder_name)
        if params['task'] != 'Blur':
            saveImage(naive_reconstruction, "naive_reconstruction", folder_name)
        saveImage(GAN_img, "GAN_img", folder_name)
        if params['skip_csgm'] == False:
            saveImage(CSGM_img, "CSGM_optimized", folder_name)
            if params['task'] != 'Blur':
                saveImage(CSGM_BP_img, "CSGM_BP", folder_name)
        saveImage(IA_img, "IA_optimized", folder_name)
        if params['task'] != 'Blur':
            saveImage(IA_BP_img, "IA_BP", folder_name)

    # Save data to tables
    if params['task'] == 'Blur':
        if params['skip_csgm'] == False:
            saveTable(original_x, None, CSGM_img, None, IA_img, None, params['parent_path'], device)
        else:
            saveTable(original_x, None, None, None, IA_img, None, params['parent_path'], device)
            
    else:
        if params['skip_csgm'] == False:
            saveTable(original_x, naive_reconstruction, CSGM_img, CSGM_BP_img, IA_img, IA_BP_img, params['parent_path'], device)
        else:
            saveTable(original_x, naive_reconstruction, None, None, IA_img, IA_BP_img, params['parent_path'], device)
            
        