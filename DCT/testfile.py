import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from itertools import product


# def simpleMask(rows, cols, ratio):
#     mask = torch.zeros((rows, cols))
#     a = torch.tensor(list(product(range(rows), range(cols))))
#     prob = torch.tensor([1/(rows*cols)]*rows*cols)
#     idx = prob.multinomial(num_samples=int(rows*cols*ratio), replacement=False)
#     for i in a[idx]:
#         mask[i[0], i[1]] = 1
#     return mask

# def compression(img, mask):
#     return torch.mul(mask, img)

# # X
# convert_to_PIL = transforms.ToPILImage()
# convert_to_tensor = transforms.ToTensor()
# path = './Images/CelebA_HQ/000168.jpg'
# x_image = Image.open(path)
# x = torch.unsqueeze(convert_to_tensor(x_image),0)

# C = simpleMask(rows=1024, cols=1024, ratio=0.5)
# y1 = torch.mul(C,x)
# y1_image = convert_to_PIL(y1[0])
# y1_image.show()

import torch
import torch_dct as dct

convert_to_PIL = transforms.ToPILImage()
convert_to_tensor = transforms.ToTensor()
path = '../Images/CelebA_HQ/000168.jpg'
x_image = Image.open(path)
x = torch.unsqueeze(convert_to_tensor(x_image),0)
X = dct.dct_2d(x)   # DCT-II done through the last dimension
X_image = convert_to_PIL(X[0])
X_image.show()

y = dct.idct_2d(X)  # scaled DCT-III done through the last dimension
y_image = convert_to_PIL(y[0])
y_image.show()