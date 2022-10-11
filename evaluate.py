import ssl
ssl._create_default_https_context = ssl._create_unverified_context

import lpips
from torchmetrics import PeakSignalNoiseRatio

'''
    Given the original image and the estimation, perform PSNR (Peak Signal-to-Noise Ratio)
'''
def PSNR(original, estimation):
    psnr = PeakSignalNoiseRatio()
    return psnr(estimation, original).item()

'''
    Given the original image and the estimation, perform PS (Perceptual similarity)
'''
def PS(original, estimation):
    loss_fn = lpips.LPIPS(net='alex')
    traditional_loss_fn = lpips.LPIPS(net='vgg')
    return loss_fn(original, estimation).item(), traditional_loss_fn(original,estimation).item()