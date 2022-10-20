import torch
import torchvision
from torchvision import transforms
from itertools import product
from PIL import Image
import warnings
warnings.filterwarnings("ignore")

convert_to_tensor = transforms.ToTensor()
convert_to_PIL = transforms.ToPILImage()

# Original
I_PIL = Image.open('./Images/CelebA_HQ/4.jpg')
#I_PIL.show()
I = convert_to_tensor(I_PIL)

# Blurring
# def blur(img, scale, dev):
#     blur = transforms.GaussianBlur(kernel_size=(scale, scale), sigma=(dev, dev))
#     return blur(img)

# blurred_I = blur(I, scale=51, dev=9)
# blurred_I_PIL = convert_to_PIL(blurred_I)
# blurred_I_PIL.show()

# FFT Compression
def render_mask(size, ratio):
    mask = torch.zeros((size, size))
    a = torch.tensor(list(product(range(size), range(size))))
    prob = torch.tensor([1/(size*size)]*size*size)
    idx = prob.multinomial(num_samples=int(size*size*ratio), replacement=False)
    for i in a[idx]:
        mask[i[0], i[1]] = 1
    return mask

def FFT(img, mask, rev_mask):
    r = img[:, 0:1, :, :]
    g = img[:, 1:2, :, :]
    b = img[:, 2:3, :, :]
    r_fft = torch.fft.fft2(r[0][0]).float()
    g_fft = torch.fft.fft2(g[0][0]).float()
    b_fft = torch.fft.fft2(b[0][0]).float()
    r_masked = torch.multiply(r_fft, mask) + torch.multiply(r[0][0], rev_mask)
    g_masked = torch.multiply(g_fft, mask) + torch.multiply(g[0][0], rev_mask)
    b_masked = torch.multiply(b_fft, mask) + torch.multiply(b[0][0], rev_mask)
    return torch.cat((r_masked.unsqueeze(0), g_masked.unsqueeze(0), b_masked.unsqueeze(0))).unsqueeze(0)

def IFFT(compressed_img, mask, rev_mask):
    r = compressed_img[:, 0:1, :, :]
    g = compressed_img[:, 1:2, :, :]
    b = compressed_img[:, 2:3, :, :]
    r_ifft = torch.fft.ifft2(r[0][0]).float()
    g_ifft = torch.fft.ifft2(g[0][0]).float()
    b_ifft = torch.fft.ifft2(b[0][0]).float()
    r_masked = torch.multiply(r_ifft, mask) + torch.multiply(r[0][0], rev_mask)
    g_masked = torch.multiply(g_ifft, mask) + torch.multiply(g[0][0], rev_mask)
    b_masked = torch.multiply(b_ifft, mask) + torch.multiply(b[0][0], rev_mask)
    return torch.cat((r_masked.unsqueeze(0), g_masked.unsqueeze(0), b_masked.unsqueeze(0)))

img = torch.unsqueeze(I, 0)
mask = render_mask(1024, 0.7)
rev_mask = mask.clone()
rev_mask[mask==0] = 1
rev_mask[mask==1] = 0

compressed_img = FFT(img, mask, rev_mask)
y_pil = convert_to_PIL(compressed_img[0])
y_pil.show()
naive_reconstruction = IFFT(compressed_img, mask, rev_mask)
x_pil = convert_to_PIL(naive_reconstruction[0])
x_pil.show()

# def compress_FFT_t(X, mask):
#     shape = mask.shape
#     mask = mask.view(-1)
#     R = torch.zeros_like(mask)
#     R[mask == 1] = X[:, 0]
#     R = R.reshape(shape)
#     G = torch.zeros_like(mask)
#     G[mask == 1] = X[:, 1]
#     G = G.reshape(shape)
#     B = torch.zeros_like(mask)
#     B[mask == 1] = X[:, 2]
#     B = B.reshape(shape)
#     r = torch.irfft(R, signal_ndim=2, normalized = True, onesided=False)
#     g = torch.irfft(G, signal_ndim=2, normalized = True, onesided=False)
#     b = torch.irfft(B, signal_ndim=2, normalized = True, onesided=False)
#     x = torch.cat((r, g, b), dim = 1)
#     return x
