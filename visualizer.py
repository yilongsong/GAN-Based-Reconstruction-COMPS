import matplotlib.pyplot as plt
import os
from torchvision import transforms
from PIL import Image
import pandas as pd
from evaluate import PSNR, PS
import torch

'''
    Save the input image to a specified folder
'''
def saveImage(img, file_name, folder_name, format='.jpg'):
    convert_to_PIL = transforms.ToPILImage()
    img = torch.clamp(img, 0, 1)
    image = convert_to_PIL(img[0])
    path = folder_name+'/'
    # Create folder if folder doesn't already exist
    if not os.path.exists(path):
            os.makedirs(path)
    image.save(path+file_name+format)
    return image

'''
    Create and save the plot
'''
def savePlot(CSGM_data, IA_data, folder_name):
    print('Processing data to create plots and tables....')
    IA_x_axis = [i + len(CSGM_data[0]) for i in range(len(IA_data[0]))]  
    plt.clf() 
    plt.plot(CSGM_data[0], CSGM_data[1], color='red')
    plt.plot(IA_x_axis, IA_data[1], color='blue')
    plt.xlabel('iteration #', fontsize=14)
    plt.ylabel('loss', fontsize=14)
    plt.grid(True)
    path = folder_name+'/plot.png'
    plt.savefig(path)

'''
    Crate and save the table
'''
def saveTable(original, Bicubic, CSGM, CSGM_BP, IA, IA_BP, folder_name, device):
    # Bicubic
    bicubic_psnr = PSNR(original[0], Bicubic[0], device)
    bicubic_ps, bicubic_ps_t = PS(original[0], Bicubic[0], device)
    # CSGM
    csgm_psnr = PSNR(original[0], CSGM[0], device)
    csgm_ps, csgm_ps_t = PS(original[0], CSGM[0], device)
    # CSGM BP
    csgm_bp_psnr = PSNR(original[0], CSGM_BP[0], device)
    csgm_bp_ps, csgm_bp_ps_t = PS(original[0], CSGM_BP[0], device)
    # IA
    ia_psnr = PSNR(original[0], IA[0], device)
    ia_ps, ia_ps_t = PS(original[0], IA[0], device)
    # IA BP
    ia_bp_psnr = PSNR(original[0], IA_BP[0], device)
    ia_bp_ps, ia_bp_ps_t = PS(original[0], IA_BP[0], device)

    # create three dataframes
    df_psnr = pd.DataFrame({
        "Bicubic": [bicubic_psnr],
        "CSGM": [csgm_psnr],
        "CSGM-BP": [csgm_bp_psnr],
        "IA": [ia_psnr],
        "IA-BP": [ia_bp_psnr]
    })

    df_ps = pd.DataFrame({
        "Bicubic": [bicubic_ps],
        "CSGM": [csgm_ps],
        "CSGM-BP": [csgm_bp_ps],
        "IA": [ia_ps],
        "IA-BP": [ia_bp_ps]
    })

    df_ps_t = pd.DataFrame({
        "Bicubic": [bicubic_ps_t],
        "CSGM": [csgm_ps_t],
        "CSGM-BP": [csgm_bp_ps_t],
        "IA": [ia_ps_t],
        "IA-BP": [ia_bp_ps_t]
    })

    #store them as csv's
    df_psnr = df_psnr.round(decimals=3)
    df_psnr.to_csv(folder_name+'psnr.csv', mode='a', index=False, header=False)

    df_ps = df_ps.round(decimals=3)
    df_ps.to_csv(folder_name+'ps.csv', mode='a', index=False, header=False)

    df_ps_t = df_ps_t.round(decimals=3)
    df_ps_t.to_csv(folder_name+'ps_t.csv', mode='a', index=False, header=False)