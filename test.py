# def save_imag(I):
#     I = torch.clamp(I, 0, 1)
#     I_PIL = convert_to_PIL(I[0])
#     I_PIL.save('output.png')

# def load_img(dir):
#     I_PIL = Image.open(dir)
#     I = convert_to_Tensor(I_PIL).unsqueeze(0)
#     return I

# I = load_img('./Images/CelebA_HQ/004288.jpg')
# out = torch.nn.functional.interpolate(I, scale_factor=1/16, mode='bicubic', align_corners=False, antialias=True)
# save_imag(out)

import torch
import torchvision
from torchvision import transforms
from PIL import Image
convert_to_tensor = transforms.ToTensor()
convert_to_PIL = transforms.ToPILImage()
I_PIL = Image.open('./Images/CelebA_HQ/4.jpg')
blur = transforms.GaussianBlur(kernel_size=(51, 51), sigma=(9, 9))
I = convert_to_tensor(I_PIL)
blurred_I = blur(I)
blurred_I_PIL = convert_to_PIL(blurred_I)
blurred_I_PIL.show()
I_PIL.show()

# Use argparse to take in command line args
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--folder', action='store', type=str, required=True)
    # args = parser.parse_args()
    # folder_name = args.folder