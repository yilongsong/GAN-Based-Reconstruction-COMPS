import matplotlib.pyplot as plt

def savePlot(CSGM_data, IA_data):

    IA_x_axis = [i + len(CSGM_data[0]) for i in range(len(IA_data[0]))]
    
    plt.plot(CSGM_data[0], CSGM_data[1], color='red')
    plt.plot(IA_x_axis, IA_data[1], color='blue')
    plt.xlabel('iteration #', fontsize=14)
    plt.ylabel('loss', fontsize=14)
    plt.grid(True)
    path = './generated_images/result5/result.png'
    plt.savefig(path)