from evaluate import PSNR, PS
from PIL import Image
from torchvision import transforms

est_path = './generated_images/result6/BP_optimized.png'
orig_path = './generated_images/result6/original.png'

convert_to_PIL = transforms.ToPILImage()
convert_to_tensor = transforms.ToTensor()

est_pil = Image.open(est_path)
orig_pil = Image.open(orig_path)

est_t = convert_to_tensor(est_pil)
orig_t = convert_to_tensor(orig_pil)

psnr = PSNR(original=orig_t, estimation=est_t)
print(psnr)

ps1, ps2 = PS(original=orig_t, estimation=est_t)
print(ps1, ps2)