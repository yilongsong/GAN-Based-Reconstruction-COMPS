from A import A
import torch
import torch.nn as nn
from re import search
from torchvision import transforms
from PIL import Image

path = './Results/PGGAN/Bicubic_0N_8S_CSGM1800_IA300_Half/0/original_x.jpg'
convert_to_tensor = transforms.ToTensor()
x_PIL = Image.open(path)
x = torch.unsqueeze(convert_to_tensor(x_PIL), 0)
bad = A.bicubic_downsample_A(x, 1/64)
convert_to_PIL = transforms.ToPILImage()
img = torch.clamp(bad, 0, 1)
image = convert_to_PIL(img[0])
image.show()