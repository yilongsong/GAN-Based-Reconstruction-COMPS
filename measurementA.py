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

convert_to_PIL = transforms.ToPILImage()
convert_to_tensor = transforms.ToTensor()

class A():
    def __init__(self):
        pass
        
    def guassian_A(img) -> torch.tensor:
        A = torch.normal(mean=0.0, std=1/np.sqrt(img.shape[2]), size=(1,3,1024,1024)) 
        return torch.multiply(A, img)

    def bicubic_downsample_A(img, scale) -> torch.tensor:
        return torch.nn.functional.interpolate(img, scale_factor=scale, mode='bicubic', align_corners=False, antialias=True)

    # take in tensor and return upsampled tensor
    def PIL_bicubic_upsample_A(img, scale) -> torch.tensor:
        size = img.shape[2]
        img_PIL = convert_to_PIL(img[0])
        new_img = img_PIL.resize((int(scale*size),int(scale*size)),Image.BICUBIC)
        return convert_to_tensor(new_img).unsqueeze(0)

    def create_simple_mask(img, ratio):
        rows = img.shape[2]
        cols = img.shape[3]
        #if (self.simple_mask == None):
        mask = torch.zeros((rows, cols))
        a = torch.tensor(list(product(range(rows), range(cols))))
        prob = torch.tensor([1/(rows*cols)]*rows*cols)
        idx = prob.multinomial(num_samples=int(rows*cols*ratio), replacement=False)
        for i in a[idx]:
            mask[i[0], i[1]] = 1
        #else:
            #mask = self.simple_mask
        return mask

    def compression_A(mask, img):
        return torch.mul(mask, img)