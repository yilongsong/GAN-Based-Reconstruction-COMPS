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
    generator = ImageAdaptiveGenerator(GAN_type='PGGAN', CSGM_optimizer="SGD", x_path='./Images/CelebA_HQ/000168.jpg', A_type="Gaussian")
    CSGM_img = generator.CSGM(2000, 0.01)
    showImage(CSGM_img)
    

if __name__ == '__main__':
    main()