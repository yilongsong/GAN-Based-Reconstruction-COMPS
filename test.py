import torch
import cv2 as cv
import numpy as np
from torchvision import transforms
from PIL import Image

from measurementA import A
from PGGAN import Generator

'''
        convert_to_PIL = transforms.ToPILImage()

        # Generated Image
        A_image = convert_to_PIL(Gz[0])
        A_image.show()

        # Comprssed 
        AGz1_image = convert_to_PIL(AGz1[0])
        AGz1_image.show()

        # Low Resolution
        AGz2_image = convert_to_PIL(AGz2[0])
        AGz2_image.show()

        convert_to_tensor = transforms.ToTensor()

        __________

        # Y (down sampled)
        self.y2 = A.bicubic_downsample_A(self.x, 1/8)
        y2_image = convert_to_PIL(y2[0])
'''


def main():
    model = Config(GAN_type, './Images/CelebA_HQ/000168.jpg')

if __name__ == '__main__':
    pass