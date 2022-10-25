'''
    Where we implement measurement metricies A

    https://discuss.pytorch.org/t/pytorch-pil-to-tensor-and-vice-versa/6312
'''
import torch
import cv2 
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from itertools import product
from torchvision import transforms
import DCT.torch_dct as dct

convert_to_PIL = transforms.ToPILImage()
convert_to_tensor = transforms.ToTensor()

class A():
    def __init__(self):
        pass
        
    def bicubic_downsample_A(img, scale) -> torch.tensor:
        return torch.nn.functional.interpolate(img, scale_factor=scale, mode='bicubic', align_corners=False, antialias=True)

    def render_mask(img, ratio):
        size = img.shape[2]
        mask = torch.zeros((size, size))
        a = torch.tensor(list(product(range(size), range(size))))
        prob = torch.tensor([1/(size*size)]*size*size)
        idx = prob.multinomial(num_samples=int(size*size*ratio), replacement=False)
        for i in a[idx]:
            mask[i[0], i[1]] = 1
        mask = mask.unsqueeze(0).unsqueeze(0).unsqueeze(4)
        return torch.cat((mask, mask), 4)

    def simple_compression_A(mask, img):
        return torch.mul(mask, img)

    def dct_compression_A(img, mask):
        return dct.idct_2d(dct.dct_2d(img) * mask)  # DCT-II done through the last dimension

    def idct_compression_A(img):
        return -1 # scaled DCT-III done through the last dimension

    def fft_compression_A(x, mask):
        x_fft_r = torch.view_as_real(torch.fft.fft2(x[:, 0:1, :, :], norm='ortho')).view(1,-1)
        x_fft_g = torch.view_as_real(torch.fft.fft2(x[:, 1:2, :, :], norm='ortho')).view(1,-1)
        x_fft_b = torch.view_as_real(torch.fft.fft2(x[:, 2:3, :, :], norm='ortho')).view(1,-1)
        mask = mask.view(-1)
        x_masked_r = x_fft_r[:, mask==1]
        x_masked_g = x_fft_g[:, mask==1]
        x_masked_b = x_fft_b[:, mask==1]
        y = torch.cat((x_masked_r.unsqueeze(1), x_masked_g.unsqueeze(1), x_masked_b.unsqueeze(1)), 1)
        return y

    def ifft_compression_A(y, mask):
        shape = mask.shape
        mask = mask.view(-1)
        r, g, b = torch.zeros_like(mask), torch.zeros_like(mask), torch.zeros_like(mask)
        r[mask == 1], g[mask == 1], b[mask == 1] = y[:, 0], y[:, 1], y[:, 2]
        r, g, b = r.reshape(shape), g.reshape(shape), b.reshape(shape)
        y_ifft_r = torch.fft.ifft2(torch.view_as_complex(r), norm='ortho').float()
        y_ifft_g = torch.fft.ifft2(torch.view_as_complex(g), norm='ortho').float()
        y_ifft_b = torch.fft.ifft2(torch.view_as_complex(b), norm='ortho').float()
        naive_x = torch.cat((y_ifft_r, y_ifft_g, y_ifft_b), 1)
        return naive_x

    def blur_A(img):
        blur = transforms.GaussianBlur(kernel_size=(51,51), sigma=(9,9))
        return blur(img[0])