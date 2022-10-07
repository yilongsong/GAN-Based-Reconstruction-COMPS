'''
x = torch.randn(200)
x = x[None, :]
x = x[None, :]
X = dct.dct_3d(x)   # DCT-II done through the last dimension
y = dct.idct_3d(X)  # scaled DCT-III done through the last dimension
#print((torch.abs(x - y)).sum())
#print((torch.abs(x - y)).sum() < 1e-10)
#assert (torch.abs(x - y)).sum() < 1e-10  # x == y within numerical tolerance

path = "../Images/CelebA_HQ/000168.jpg"
img = Image.open(path)
convert_to_PIL = transforms.ToPILImage()
convert_to_tensor = transforms.ToTensor()

x = convert_to_tensor(img) # good image
'''