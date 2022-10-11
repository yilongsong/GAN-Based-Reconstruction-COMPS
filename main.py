'''
    Configure model and run each step to get results
'''
from model import ImageAdaptiveGenerator 
from torchvision import transforms
from visualizer import savePlot, saveImage, saveTable

def main():
    folder_name = 'result10'
    # Hussein et al.'s method
    generator = ImageAdaptiveGenerator(GAN_type='PGGAN', CSGM_optimizer="ADAM", x_path='./Images/CelebA_HQ/004288.jpg', \
                                        A_type="DCT_Compression", IA_optimizer_z="ADAM", IA_optimizer_G="ADAM", \
                                        scale=0.5, result_folder_name=folder_name)
    
    saveImage(generator.x, "original", folder_name)
    saveImage(generator.y, "downsampled", folder_name)

    # Naive Reconstruction through inverse A
    original, BICUBIC_img = generator.Naive()
    saveImage(BICUBIC_img, "BICUBIC_naive", folder_name)
    
    # CSGM 
    CSGM_img, original1, CSGM_data = generator.CSGM(csgm_iteration_number=50, csgm_learning_rate=0.1)
    saveImage(original1, "CSGM_orginal", folder_name)
    saveImage(CSGM_img, "CSGM_optimized", folder_name)

    # CSGM_BP
    CSGMBP_img = generator.BP()
    saveImage(CSGMBP_img, "CSGM_BP", folder_name)

    # IA
    IA_img, original2, IA_data = generator.IA(IA_iteration_number=50, IA_z_learning_rate=0.0001, IA_G_learning_rate=0.001)
    saveImage(IA_img, "IA_optimized", folder_name)

    # IA_BP
    IABP_img = generator.BP()
    saveImage(IABP_img, "IA_BP", folder_name)

    # # Save data as line graphs
    savePlot(CSGM_data, IA_data, folder_name)

    # # Save data as a table
    saveTable(original, BICUBIC_img, CSGM_img, CSGMBP_img, IA_img, IABP_img, folder_name)

if __name__ == '__main__':
    main()