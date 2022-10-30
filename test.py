# from visualizer import get_confidence, get_summary
# get_summary('./Results/PGGAN/Bicubic_0N_8S_CSGM1800_IA300_Half/')
# get_confidence('./Results/PGGAN/Bicubic_0N_16S_CSGM1800_IA300_Half/')

import torch
from torchvision import transforms
from PIL import Image
from A import A

convert_to_tensor = transforms.ToTensor()
x_PIL = Image.open('./Images/CelebA_HQ/0.jpg')
x = torch.unsqueeze(convert_to_tensor(x_PIL), 0)

mask = A.render_mask(x, 0.5)
naive = A.ifft_compression_A(A.fft_compression_A(x, mask), mask)

convert_to_PIL = transforms.ToPILImage()
img = torch.clamp(naive, 0, 1)
image = convert_to_PIL(img[0])
image.show()
