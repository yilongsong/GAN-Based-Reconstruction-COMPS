'''
    Configure model and run each step to get results
'''
from model import ImageAdaptiveGenerator 
from torchvision import transforms
from image_saver import saveImage

def main():
    # Hussein et al.'s method
    generator = ImageAdaptiveGenerator(GAN_type='PGGAN', CSGM_optimizer="ADAM", x_path='./Images/CelebA_HQ/001349.jpg', \
                                        A_type="Bicubic_Downsample", IA_optimizer_z="ADAM", IA_optimizer_G="ADAM")
    
    saveImage(generator.x, "target_original")
    saveImage(generator.y, "target_downsampled")
    
    # CSGM
    CSGM_img, original1 = generator.CSGM(csgm_iteration_number=1000, csgm_learning_rate=0.1)
    saveImage(original1, "csgm_orginal")
    saveImage(CSGM_img, "csgm_optimized")

    # IA
    IA_img, original2 = generator.IA(IA_iteration_number=200, IA_z_learning_rate=0.0001, IA_G_learning_rate=0.001)
    saveImage(IA_img, "ia_optimized")

    # BP
    BP_img = generator.BP()
    saveImage(BP_img, "bp_optimized")

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