# import torch
# import torchvision
# from torchvision import transforms
# from itertools import product
# from PIL import Image
# import warnings
# import numpy as np
# warnings.filterwarnings("ignore")

# convert_to_tensor = transforms.ToTensor()
# convert_to_PIL = transforms.ToPILImage()

# # Original
# I_PIL = Image.open('./Images/CelebA_HQ/0.jpg')
# I = convert_to_tensor(I_PIL)
# x = torch.unsqueeze(I, 0)

# from A import A
# y = A.blur_A(x, 9)
# y = torch.clamp(y, 0, 1)
# y = convert_to_PIL(y[0])
# y.show()
# x = 1

import numpy as np
import csv


def get_average_results(folder):
    def get_info(path):
        file = csv.reader(open(path, 'r'))
        count = 0
        sum = [0.0, 0.0, 0.0, 0.0, 0.0]
        for row in file:
            count += 1
            for i in range(len(row)):
                sum[i] += float(row[i])
        avg = [round(sum[i]/count,3) for i in range(len(sum))]
        print(avg)
    
    get_info(folder+'ps_t.csv')
    get_info(folder+'ps.csv')
    get_info(folder+'psnr.csv')
    print('********************')

get_average_results('./Results/PGGAN/Bicubic_0N_8S_CSGM1800_IA300_Object/')
get_average_results('./Results/PGGAN/Bicubic_0N_8S_CSGM1800_IA300_DarkHair/')
get_average_results('./Results/PGGAN/Bicubic_0N_8S_CSGM1800_IA300_LightHair/')
get_average_results('./Results/PGGAN/Bicubic_0N_8S_CSGM1800_IA300_NoHair/')
get_average_results('./Results/PGGAN/Bicubic_0N_8S_CSGM1800_IA300_DarkSkin/')
get_average_results('./Results/PGGAN/Bicubic_0N_8S_CSGM1800_IA300_LightSkin/')
