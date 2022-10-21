'''
    Configure model and run each step to get results
    https://pynative.com/python-copy-files-and-directories/
'''

import shutil
from re import search, split
from os import listdir, remove
from os.path import isfile, join
from model import run_model

def reset_weights():
    src_path = './saved_weights/100_celeb_hq_network-snapshot-010403.pth'
    dst_path = './weights/100_celeb_hq_network-snapshot-010403.pth'
    remove(dst_path)
    shutil.copy(src_path, dst_path)
    print('Weights Copied')

def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    return [ atoi(c) for c in split(r'(\d+)', text) ]

def main():
    img_dir = './Images/CelebA_HQ'
    files = [f for f in listdir(img_dir) if isfile(join(img_dir, f))]
    images = [f for f in files if ('.jpg' or '.png') in f]
    images.sort(key=natural_keys)
    
    count = 0
    for img in images:
        if count < 30 or count == 15:
            count += 1
            continue
        elif count >= 101:
            break
        
        print('Start reconstruction on ' + img)
        parent_path = './Results/Bicubic_10N_16S/'
        img_folder = search(r'\d+', img).group()
        run_model(img=img_dir+'/'+img, parent_path=parent_path, img_folder=img_folder)
        reset_weights()
        count += 1

if __name__ == '__main__':
    main()