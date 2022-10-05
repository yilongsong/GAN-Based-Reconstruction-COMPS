'''
    Configure model and run each step to get results
'''
from model import ImageAdaptiveGenerator 
from torchvision import transforms
from image_saver import saveImage
from plot import savePlot

def main():
    folder_name = 'TEST'
    # Hussein et al.'s method
    generator = ImageAdaptiveGenerator(GAN_type='PGGAN', CSGM_optimizer="ADAM", x_path='./Images/CelebA_HQ/001743.jpg', \
                                        A_type="Bicubic_Downsample", IA_optimizer_z="ADAM", IA_optimizer_G="ADAM", \
                                        result_folder_name=folder_name)
    
    saveImage(generator.x, "original", folder_name)
    saveImage(generator.y, "downsampled", folder_name)
    
    # CSGM (1800)
    CSGM_img, original1, CSGM_data = generator.CSGM(csgm_iteration_number=1800, csgm_learning_rate=0.1)
    saveImage(original1, "CSGM_orginal", folder_name)
    saveImage(CSGM_img, "CSGM_optimized", folder_name)

    # IA (300)
    IA_img, original2, IA_data = generator.IA(IA_iteration_number=300, IA_z_learning_rate=0.0001, IA_G_learning_rate=0.001)
    saveImage(IA_img, "IA_optimized", folder_name)

    # BP
    BP_img = generator.BP()
    saveImage(BP_img, "BP_optimized", folder_name)

    # Save data as line graphs
    savePlot(CSGM_data, IA_data)

if __name__ == '__main__':
    main()