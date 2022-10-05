'''
    Where we implement measurement metricies A

    https://discuss.pytorch.org/t/pytorch-pil-to-tensor-and-vice-versa/6312
'''
import torch
import cv2 
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

from torchvision import transforms

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
        convert_to_PIL = transforms.ToPILImage()
        convert_to_tensor = transforms.ToTensor()
        size = img.shape[2]
        img_PIL = convert_to_PIL(img[0])
        new_img = img_PIL.resize((int(scale*size),int(scale*size)),Image.BICUBIC)
        return convert_to_tensor(new_img).unsqueeze(0)