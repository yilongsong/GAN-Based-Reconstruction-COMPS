import torch
import cv2 
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms

# def cv_bicubic_downsample_A(img):
#         np_img = img
#         resized = np.zeros((3,128,128))
#         for ch in range(3):
#             resized_ch = cv2.resize(np_img[ch], dsize=(128,128), interpolation=cv2.INTER_NEAREST)
#             resized[ch] = resized_ch
#         img = torch.from_numpy(resized)
#         return img.unsqueeze(0)

# def cv_bicubic_upsample_A(img):
#         np_img = img.numpy()[0]
#         print(np.shape(np_img))
#         resized = np.zeros((3,1024,1024))
#         for ch in range(3):
#             resized_ch = cv2.resize(np_img[ch], dsize=(1024,1024), interpolation=cv2.INTER_NEAREST)
#             resized[ch] = resized_ch
#         img = torch.from_numpy(resized)
#         return img.unsqueeze(0)

#convert_to_tensor = transforms.ToTensor()
x_PIL = Image.open('./Images/CelebA_HQ/005152.jpg')
# #x = torch.unsqueeze(convert_to_tensor(x_PIL), 0)
# #y = cv_bicubic_downsample_A(x)
# x = np.array(x_PIL).transpose(2,0,1)
# y = cv_bicubic_downsample_A(x)
# x_hat = cv_bicubic_upsample_A(y)

# def showImage(img):
#         convert_to_PIL = transforms.ToPILImage()
#         image = convert_to_PIL(img[0])
#         image.show()
#         return image

# showImage(x)
# showImage(y)
# showImage(x_hat)

new_img = x_PIL.resize((64,64),Image.BICUBIC)
new_img.show()