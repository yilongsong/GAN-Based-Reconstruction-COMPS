import os

from torchvision import transforms
from PIL import Image

def saveImage(img, file_name, format='.png'):
        convert_to_PIL = transforms.ToPILImage()
        image = convert_to_PIL(img[0])
        path = './generated_images/result/'
        image.save(path+file_name+format)
        return image