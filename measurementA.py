'''
    Where we implement measurement metrecies A
'''
import torch
import cv2 
import numpy as np
import matplotlib.pyplot as plt

class A():
    def __init__(self):
        pass

    def guassian_A(img) -> torch.tensor:
        A = torch.normal(mean=0.0, std=1/np.sqrt(img.shape[2]), size=(1,3,1024,1024)) 
        return torch.multiply(A, img)

    def bicubic_downsample_A(img, scale) -> torch.tensor:
        return torch.nn.functional.interpolate(img, scale_factor=scale, mode='bicubic') 
    
    # equivalent to A_dag(img)
    #def bicubic_upsample_A(img, scale) -> torch.tensor:
    #    return torch.nn.functional.interpolate(img, scale_factor=scale, mode='bicubic')

    def cv_bicubic_downsample_A(img):
        np_img = img.numpy()[0]
        resized = np.zeros((3,512,512))
        for ch in range(3):
            resized_ch = cv2.resize(np_img[ch], dsize=(512,512), interpolation=cv2.INTER_CUBIC)
            resized[ch] = resized_ch
        img = torch.from_numpy(resized)
        return img.unsqueeze(0)