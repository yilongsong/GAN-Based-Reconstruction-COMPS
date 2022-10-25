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

# # Blurring
# def blur(img, scale, dev):
#     blur = transforms.GaussianBlur(kernel_size=(scale, scale), sigma=(dev, dev))
#     return blur(img[0])

# y = blur(x, scale=51, dev=9)
# PILy = convert_to_PIL(y)
# # PILy.show()

import DCT.torch_dct as dct
from A import A


I_PIL = Image.open('./Images/CelebA_HQ/4.jpg')
I = convert_to_tensor(I_PIL)
x = torch.unsqueeze(I, 0)

mask = A.render_mask(x, 0.7)
y = A.fft_compression_A(x, mask)
naive = A.ifft_compression_A(y, mask)

# print(y.shape)
# print(naive.shape)


naivePIL = convert_to_PIL(torch.clamp(naive[0], 0, 1))
naivePIL.show()