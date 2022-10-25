import torch
import torchvision
from torchvision import transforms
from itertools import product
from PIL import Image
import warnings
import numpy as np
warnings.filterwarnings("ignore")

convert_to_tensor = transforms.ToTensor()
convert_to_PIL = transforms.ToPILImage()

# Original
I_PIL = Image.open('./Images/CelebA_HQ/4.jpg')
I = convert_to_tensor(I_PIL)
x = torch.unsqueeze(I, 0)
