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
    parser = argparse.ArgumentParser()
    parser.add_argument('--GAN', type=str, required=True)
    parser.add_argument('--scale', type=int, required=True)
    parser.add_argument('--noise', type=int, required=True)
    parser.add_argument('--task', type=str, required=True)
    parser.add_argument('--save_images', action='store_true')
    args = parser.parse_args()

    GANs = ['PGGAN', 'DCGAN']
    scales = [4, 8, 16, 32]
    noises = [0, 10, 40]
    tasks = ['Naive_Compression', 'DCT_Compression', 'Bicubic']

    if args.GAN not in GANs:
        print('ERROR: GAN not found')
    elif args.scale not in scales:
        print('ERROR: scale needs to be 4, 8, 16, or 32')
    elif args.noise not in noises:
        print('ERROR: noise level needs to be 0, 10, or 40')
    elif args.task not in tasks:
        print('ERROR: not a valid task')

    params = {
        'GAN': args.GAN,
        'scale': args.scale,
        'noise_level': args.noise,
        'A_type': args.task,
        'parent_path': './Results/' + args.GAN + '/' + args.task + '_' \
            + str(args.noise) + 'N_' + str(args.scale) + 'S/',
        'save_images': args.save_images
    }

    if args.GAN == 'PGGAN':
        params['img_dir'] = './Images/CelebA_HQ'
        params['weights_path'] = 'PGGAN_weights.pth'
        params['CSGM_itr'] = 1800
        params['IA_G_learning_rate'] = 0.001
    else:
        params['img_dir'] = './Images/CelebA'
        params['weights_path'] = 'DCGAN_weights.pth'
        params['CSGM_itr'] = 1600
        params['IA_G_learning_rate'] = 0.0001

    return params

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

    # Run the model on each image
    for img in images:
        print('Start reconstruction on ' + img)
        run_model(img, params)
        reset_weights(params['weights_path'])

if __name__ == '__main__':
    main()