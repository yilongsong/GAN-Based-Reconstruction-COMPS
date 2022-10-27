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
    Crate and save the table
'''
def saveTable(original, Naive, CSGM, CSGM_BP, IA, IA_BP, folder_name, device):
    print('Processing data to create tables....')
    # dicts for storing data
    psnr, ps, ps_t = {}, {}, {}
    
    # Naive
    if Naive != None:
        naive_psnr = PSNR(original[0], Naive[0], device)
        naive_ps, naive_ps_t = PS(original[0], Naive[0], device)
        psnr['Native'], ps['Native'], ps_t['Native'] = [naive_psnr], [naive_ps], [naive_ps_t]
    # CSGM
    if CSGM != None:
        csgm_psnr = PSNR(original[0], CSGM[0], device)
        csgm_ps, csgm_ps_t = PS(original[0], CSGM[0], device)
        psnr['CSGM'], ps['CSGM'], ps_t['CSGM'] = [csgm_psnr], [csgm_ps], [csgm_ps_t]
    # CSGM BP
    if CSGM_BP != None:
        csgm_bp_psnr = PSNR(original[0], CSGM_BP[0], device)
        csgm_bp_ps, csgm_bp_ps_t = PS(original[0], CSGM_BP[0], device)
        psnr['CSGM-BP'], ps['CSGM-BP'], ps_t['CSGM-BP'] = [csgm_bp_psnr], [csgm_bp_ps], [csgm_bp_ps_t]
    # IA
    if IA != None:
        ia_psnr = PSNR(original[0], IA[0], device)
        ia_ps, ia_ps_t = PS(original[0], IA[0], device)
        psnr['IA'], ps['IA'], ps_t['IA'] = [ia_psnr], [ia_ps], [ia_ps_t]
    # IA BP
    if IA_BP != None:
        ia_bp_psnr = PSNR(original[0], IA_BP[0], device)
        ia_bp_ps, ia_bp_ps_t = PS(original[0], IA_BP[0], device)
        psnr['IA-BP'], ps['IA-BP'], ps_t['IA-BP'] = [ia_bp_psnr], [ia_bp_ps], [ia_bp_ps_t]

    # create three dataframes
    df_psnr = pd.DataFrame(psnr).round(decimals=3)
    df_ps = pd.DataFrame(ps).round(decimals=3)
    df_ps_t = pd.DataFrame(ps_t).round(decimals=3)

    #store them as csv's
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
    df_psnr.to_csv(folder_name+'psnr.csv', mode='a', index=False, header=False)
    df_ps.to_csv(folder_name+'ps.csv', mode='a', index=False, header=False)
    df_ps_t.to_csv(folder_name+'ps_t.csv', mode='a', index=False, header=False)

    print('Tables Generated')