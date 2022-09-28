import torch
import matplotlib.pyplot as plt
import torchvision
from torchvision.utils import save_image

# they have an invalid certificate problem on their end
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

use_gpu = True if torch.cuda.is_available() else False

# trained on high-quality celebrity faces "celebA" dataset
# this model outputs 512 x 512 pixel images
model_high_res = torch.hub.load('facebookresearch/pytorch_GAN_zoo:hub',
                        'PGAN', model_name='celebAHQ-512',
                        pretrained=True, useGPU=use_gpu)

# this model outputs 256 x 256 pixel images
model_middle_res = torch.hub.load('facebookresearch/pytorch_GAN_zoo:hub',
                        'PGAN', model_name='celebAHQ-256',
                        pretrained=True, useGPU=use_gpu)

# this model outputs 128 x 128 pixel images
model_low_res = torch.hub.load('facebookresearch/pytorch_GAN_zoo:hub',
                        'PGAN', model_name='celeba',
                        pretrained=True, useGPU=use_gpu)

# this model outputs 128 x 128 pixel images
model_DTD = torch.hub.load('facebookresearch/pytorch_GAN_zoo:hub',
                        'PGAN', model_name='DTD',
                        pretrained=True, useGPU=use_gpu)


# input noise from a latent space is N * 521, where N = # of output images
# presumably this will allow the system to carry out the process in parallel for each image
num_images = 1
noise, _ = model_high_res.buildNoiseData(num_images) 
with torch.no_grad():
    generated_images = model_high_res.test(noise)

torch.save(generated_images, 'generated_images')
save_image(generated_images[0], 'generated.png')


# plot these images using torchvision and matplotlib
grid = torchvision.utils.make_grid(generated_images.clamp(min=-1, max=1), scale_each=True, normalize=True)
plt.imshow(grid.permute(1, 2, 0).cpu().numpy())
plt.show()