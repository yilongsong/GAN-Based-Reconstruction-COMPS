'''
    Configure model and run each step to get results
'''
import argparse
from model import ImageAdaptiveGenerator 
from visualizer import savePlot, saveImage, saveTable

def main():
    # Use argparse to take in command line args
    parser = argparse.ArgumentParser()
    parser.add_argument('--folder', action='store', type=str, required=True)
    args = parser.parse_args()
    folder_name = args.folder

    # Instanciate our IAGAN
    generator = ImageAdaptiveGenerator(
            GAN_type='PGGAN', 
            CSGM_optimizer="ADAM", 
            IA_optimizer_z="ADAM", 
            IA_optimizer_G="ADAM",
            x_path='./Images/CelebA_HQ/002826.jpg',
            A_type="Bicubic_Downsample", 
            noise_level=40/255,
            scale=1/16, 
            result_folder_name=folder_name)
    
    # Orginal "good" image x and degraded image y
    original_x = generator.x
    degraded_y = generator.y
    saveImage(original_x, "original_x", folder_name)
    saveImage(degraded_y, "degraded_y", folder_name)

    # Naive Reconstruction through pseudo-inverse A
    naive_reconstruction = generator.Naive()
    saveImage(naive_reconstruction, "naive_reconstruction", folder_name)

    # Image produced by GAN with the initial z
    GAN_img = generator.GAN()
    saveImage(GAN_img, "GAN_img", folder_name)
    
    # CSGM 
    CSGM_img, CSGM_data = generator.CSGM(csgm_iteration_number=10, csgm_learning_rate=0.1)
    saveImage(CSGM_img, "CSGM_optimized", folder_name)

    # CSGM-BP
    CSGM_BP_img = generator.BP()
    saveImage(CSGM_BP_img, "CSGM_BP", folder_name)

    # IA
    IA_img, IA_data = generator.IA(IA_iteration_number=10, IA_z_learning_rate=0.0001, IA_G_learning_rate=0.001)
    saveImage(IA_img, "IA_optimized", folder_name)

    # IA_BP
    IA_BP_img = generator.BP()
    saveImage(IA_BP_img, "IA_BP", folder_name)

    # Save data as line graphs
    savePlot(CSGM_data, IA_data, folder_name)

    # Save data as a table
    saveTable(original_x, naive_reconstruction, CSGM_img, CSGM_BP_img, IA_img, IA_BP_img, folder_name)

if __name__ == '__main__':
    main()