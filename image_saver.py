from torchvision import transforms
from PIL import Image

def saveImage(img, file_name):
        convert_to_PIL = transforms.ToPILImage()
        image = convert_to_PIL(img[0])
        image.save("./generated_images/"+file_name)
        return image