import matplotlib.pyplot as plt
import os
from torchvision import transforms
from PIL import Image

def saveImage(img, file_name, folder_name, format='.png'):
    convert_to_PIL = transforms.ToPILImage()
    image = convert_to_PIL(img[0])
    path = './generated_images/'+folder_name+'/'
    # Create folder if folder doesn't already exist
    if not os.path.exists(path):
            os.makedirs(path)
    image.save(path+file_name+format)
    return image

def savePlot(CSGM_data, IA_data):
    IA_x_axis = [i + len(CSGM_data[0]) for i in range(len(IA_data[0]))]   
    plt.plot(CSGM_data[0], CSGM_data[1], color='red')
    plt.plot(IA_x_axis, IA_data[1], color='blue')
    plt.xlabel('iteration #', fontsize=14)
    plt.ylabel('loss', fontsize=14)
    plt.grid(True)
    path = './generated_images/result6/result.png'
    plt.savefig(path)

def saveTable(CSGM_result, CSGM_BP_result, IA_result, IA_BP_result):
    pass