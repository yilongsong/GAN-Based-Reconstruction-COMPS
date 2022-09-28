import torch
import cv2 
import numpy as np
import matplotlib.pyplot as plt
import os

#device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
img = cv2.imread("../Preparation/CelebA_HQ/000168.jpg", cv2.IMREAD_ANYCOLOR)
img_tensor = torch.from_numpy(img)
#cv2.imshow("Test Image from CelebA-HQ", img)
#cv2.waitKey(0)

# Gaussian Measurement A
mu, sigma = 0, 1/np.sqrt(1024)
A = np.random.normal(mu, sigma, (1024, 1024, 3))
apply_A = lambda img : np.multiply(A, img)
#cv2.imshow("Compressed Test Image", apply_A(img))
#cv2.waitKey(0)