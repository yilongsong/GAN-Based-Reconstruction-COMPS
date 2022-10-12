import ssl
ssl._create_default_https_context = ssl._create_unverified_context

import lpips
from torchmetrics import PeakSignalNoiseRatio

'''
    Given the original image and the estimation, perform PSNR (Peak Signal-to-Noise Ratio)
'''
def PSNR(original, estimation, device):
    psnr = PeakSignalNoiseRatio().to(device)
    return psnr(estimation, original).item()

'''
    Given the original image and the estimation, perform PS (Perceptual similarity)
'''
def PS(original, estimation, device):
    loss_fn = lpips.LPIPS(net='alex').to(device)
    traditional_loss_fn = lpips.LPIPS(net='vgg').to(device)
    return loss_fn(original, estimation).item(), traditional_loss_fn(original,estimation).item()