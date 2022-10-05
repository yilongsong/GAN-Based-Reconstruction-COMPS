import os

from torchvision import transforms
from PIL import Image

def saveImage(img, file_name, folder_name, format='.png'):
        convert_to_PIL = transforms.ToPILImage()
        image = convert_to_PIL(img[0])
        path = './generated_images/'+folder_name+'/'
        # Create folder if folder doesn't already exist
        if not os.path.exists(path):
                os.makedirs(path)
        image.save(path+file_name+format)
        return image