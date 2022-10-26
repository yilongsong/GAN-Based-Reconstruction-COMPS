'''
    Configure model and run each step to get results
    https://pynative.com/python-copy-files-and-directories/
'''

import shutil
import argparse
from re import split
from os import listdir, remove
from os.path import isfile, join
from model import run_model

def parse_args():
    # parse command line args
    parser = argparse.ArgumentParser()
    parser.add_argument('--GAN', type=str, required = True, choices=['PGGAN'], help='type of pre-trained GAN: PGGAN')
    parser.add_argument('--scale', type=int, required=False, choices=[4,8,16,32], help='scale of Bicubic: 4, 8, 16, or 32 for scale')
    parser.add_argument('--ratio', type=float, required=False, choices=[0.1,0.3,0.5,0.7], help='ratio of FFT: 0.1, 0.3, 0.5, or 0.7')
    parser.add_argument('--noise', type=int, required=True, choices=[0,10,40], help='noise level of y: 0, 10, or 40')
    parser.add_argument('--task', type=str, required=True, choices=['FFT', 'Bicubic', 'Blur'], help='task to be performed: FFT, Bicubic, or Blur')
    parser.add_argument('--csgm_itr', type=int, required=True, choices=[0,500,1000,1500,1800], help='number of iterations for CSGM')
    parser.add_argument('--ia_itr', type=int, required=True, choices=[300,500,600,800,1100,1300,2100,2300], help='number of iterations for IA')
    parser.add_argument('--test_folder', type=str, required=True, choices=['Whole', 'Object', 'DarkHair', 'LightHair', 'NoHair', 'DarkSkin', 'LightSkin'], help='folder of images that you want to run this model on')
    parser.add_argument('--save_images', action='store_true', help='include this argument to save the result images')
    args = parser.parse_args()

    # create the name of result folder
    folder = './Results/' + args.GAN + '/' + args.task + '_' + str(args.noise) + 'N'
    if args.task != 'Blur':
        folder += '_' + (str(args.scale) + 'S' if args.scale != None else str(args.ratio) + 'R')
    folder += ('_CSGM' + str(args.csgm_itr))
    folder += ('_IA' + str(args.ia_itr))
    folder += ('_' + args.test_folder + '/')

    # create a dict of parameters
    params = {
        'GAN': args.GAN,
        'rate': 1/args.scale if args.scale != None else args.ratio,
        'noise_level': args.noise/255,
        'task': args.task,
        'CSGM_itr': args.csgm_itr,
        'IA_itr': args.ia_itr,
        'parent_path': folder,
        'test_folder': args.test_folder,
        'skip_csgm': True if args.csgm_itr == 0 else False,
        'save_images': args.save_images
    }

    # configure parameters based on GAN type
    if args.GAN == 'PGGAN':
        params['img_dir'] = './Images/CelebA_HQ'
        params['weights_path'] = 'PGGAN_weights.pth'
        params['CSGM_lr'] = 0.1
        params['IA_z_lr'] = 0.0001
        params['IA_G_lr'] = 0.001

    return params

def get_test_folder(folder):
    if folder == 'Whole':
        return [str(i) for i in range(0, 100)]
    elif folder == 'Object':
        return ['7', '11', '45', '47', '48', '61', '66', '69', '73', '74']
    elif folder == 'DarkHair':
        return ['4', '5', '6', '10', '15', '16', '18', '50', '51', '55']
    elif folder == 'LightHair':
        return ['0', '1', '3', '7', '12', '13', '46', '62', '76', '92']
    elif folder == 'NoHair':
        return ['15', '30', '45', '47', '59', '89']
    elif folder == 'DarkSkin':
        return ['7', '31', '38', '39', '42', '50', '90', '100', '101', '102']
    elif folder == 'LightSkin':
        return [str(i) for i in range(0, 10)]

def reset_weights(weights_path):
    src_path = './saved_weights/' + weights_path
    dst_path = './weights/' + weights_path
    remove(dst_path)
    shutil.copy(src_path, dst_path)
    print('Weights Copied')

def natural_keys(text):
    atoi = lambda x: int(x) if x.isdigit() else x
    return [ atoi(c) for c in split(r'(\d+)', text) ]

def main():
    # Read and get command line args
    params = parse_args()
    
    # Create a list of training/testing images
    img_dir = params['img_dir']
    files = [f for f in listdir(img_dir) if isfile(join(img_dir, f))]
    images = [f for f in files if ('.jpg' or '.png') in f]
    images.sort(key=natural_keys)

    # get test_folder
    test_folder = get_test_folder(params['test_folder'])

    # Run the model on each image
    num_img_saved = 0
    for img in images:
        print('Start reconstruction on ' + img)
        if num_img_saved >= 5:
            params['save_images'] = False
        if img[:-4] in test_folder:
            run_model(img, params)
            reset_weights(params['weights_path'])
            num_img_saved += 1

if __name__ == '__main__':
    main()