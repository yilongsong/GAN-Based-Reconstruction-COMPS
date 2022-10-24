import torch
import torchvision
from torchvision import transforms
from itertools import product
from PIL import Image
import warnings
warnings.filterwarnings("ignore")

convert_to_tensor = transforms.ToTensor()
convert_to_PIL = transforms.ToPILImage()

# Original
I_PIL = Image.open('./Images/CelebA_HQ/4.jpg')
I = convert_to_tensor(I_PIL)
x = torch.unsqueeze(I, 0)

# FFT Compression
from A import A
mask = A.render_mask(x, 0.5)
signal = A.fft_compression_A(x, mask)
naive = A.ifft_compression_A(signal)

naive = torch.clamp(naive, 0, 1)
PILnaive = convert_to_PIL(naive[0])
PILnaive.show()

# Blurring
# def blur(img, scale, dev):
#     blur = transforms.GaussianBlur(kernel_size=(scale, scale), sigma=(dev, dev))
#     return blur(img)

# blurred_I = blur(I, scale=51, dev=9)
# blurred_I_PIL = convert_to_PIL(blurred_I)
# blurred_I_PIL.show()