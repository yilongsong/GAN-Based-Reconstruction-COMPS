'''
    Configure model and run each step to get results
'''
from model import ImageAdaptiveGenerator 
from torchvision import transforms

def showImage(img):
        convert_to_tensor = transforms.ToPILImage()
        image = convert_to_tensor(img[0])
        image.show()
        return image

def main():
    generator = ImageAdaptiveGenerator(GAN_type='PGGAN', CSGM_optimizer="ADAM", x_path='./Images/CelebA_HQ/028580.jpg', A_type="Bicubic_Downsample")
    CSGM_img, org = generator.CSGM(2000, 0.1)
    showImage(CSGM_img)
    showImage(org)
    #generator.BP()

if __name__ == '__main__':
    main()