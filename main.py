'''
    Configure model and run each step to get results
'''
from model import ImageAdaptiveGenerator 
from torchvision import transforms

def saveImage(img, file_name):
        convert_to_PIL = transforms.ToPILImage()
        image = convert_to_PIL(img[0])
        image.save("./generated_images/"+file_name)
        return image

def main():
    # Hussein et al.'s method
    generator = ImageAdaptiveGenerator(GAN_type='PGGAN', CSGM_optimizer="ADAM", x_path='./Images/CelebA_HQ/001349.jpg', \
                                        A_type="Bicubic_Downsample", IA_optimizer_z="ADAM", IA_optimizer_G="ADAM")
    
    saveImage(generator.x, "target_original.png")
    saveImage(generator.y, "target_downsampled.png")
    
    # CSGM
    CSGM_img, original1 = generator.CSGM(csgm_iteration_number=1000, csgm_learning_rate=0.1)
    saveImage(original1, "csgm_orginal,png")
    saveImage(CSGM_img, "csgm_optimized.png")

    # IA
    IA_img, original2 = generator.IA(IA_iteration_number=200, IA_z_learning_rate=0.0001, IA_G_learning_rate=0.001)
    saveImage(IA_img, "ia_optimized.png")

    # BP
    BP_img = generator.BP()
    saveImage(BP_img, "bp_optimized.png")

    # We need to get the original weights every time we run IA step!!!!!!

    # # Our method
    # generator = ImageAdaptiveGenerator(GAN_type='PGGAN', CSGM_optimizer="ADAM", x_path='./Images/CelebA_HQ/028580.jpg', \
    #                                     A_type="Bicubic_Downsample", IA_optimizer_z="ADAM", IA_optimizer_G="ADAM")
    
    # # IA
    # IA_img, original2 = generator.IA(IA_iteration_number=1, IA_z_learning_rate=0.0001, IA_G_learning_rate=0.001)
    # showImage(original2)
    # showImage(IA_img)

    # # BP
    # BP_img = generator.BP()
    # showImage(BP_img)

if __name__ == '__main__':
    main()