import matplotlib.pyplot as plt
import os
from torchvision import transforms
from PIL import Image
import pandas as pd
from evaluate import PSNR, PS

def saveImage(img, file_name, folder_name, format='.png'):
    convert_to_PIL = transforms.ToPILImage()
    image = convert_to_PIL(img[0])
    path = './generated_images/'+folder_name+'/'
    # Create folder if folder doesn't already exist
    if not os.path.exists(path):
            os.makedirs(path)
    image.save(path+file_name+format)
    return image

def savePlot(CSGM_data, IA_data, folder_name):
    print('Processing data to create plots and tables....')
    IA_x_axis = [i + len(CSGM_data[0]) for i in range(len(IA_data[0]))]   
    plt.plot(CSGM_data[0], CSGM_data[1], color='red')
    plt.plot(IA_x_axis, IA_data[1], color='blue')
    plt.xlabel('iteration #', fontsize=14)
    plt.ylabel('loss', fontsize=14)
    plt.grid(True)
    path = './generated_images/'+folder_name+'/result.png'
    plt.savefig(path)

def saveTable(original, Bicubic, CSGM, CSGM_BP, IA, IA_BP, folder_name):
    # Bicubic
    bicubic_psnr = PSNR(original[0], Bicubic[0])
    bicubic_ps = PS(original[0], Bicubic[0])
    # CSGM
    csgm_psnr = PSNR(original[0], CSGM[0])
    csgm_ps = PS(original[0], CSGM[0])
    # CSGM BP
    csgm_bp_psnr = PSNR(original[0], CSGM_BP[0])
    csgm_bp_ps = PS(original[0], CSGM_BP[0])
    # IA
    ia_psnr = PSNR(original[0], IA[0])
    ia_ps = PS(original[0], IA[0])
    # IA BP
    ia_bp_psnr = PSNR(original[0], IA_BP[0])
    ia_bp_ps = PS(original[0], IA_BP[0])
    # Create columns
    bicubic = [bicubic_psnr, bicubic_ps[0],bicubic_ps[1]]
    csgm = [csgm_psnr, csgm_ps[0], csgm_ps[1]]
    csgm_bp = [csgm_bp_psnr, csgm_bp_ps[0], csgm_bp_ps[1]]
    ia = [ia_psnr, ia_ps[0], ia_ps[1]]
    ia_bp = [ia_bp_psnr, ia_bp_ps[0], ia_bp_ps[1]]
    # create dataframe
    df = pd.DataFrame({
        "Bicubic": bicubic,
        "CSGM": csgm,
        "CSGM-BP": csgm_bp,
        "IA": ia,
        "IA-BP": ia_bp
    })
    #store it as csv
    path = './generated_images/'+folder_name+'/table.csv'
    df = df.round(decimals=3)
    df.to_csv(path, index=False)