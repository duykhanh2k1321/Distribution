import os
import numpy as np
import matplotlib.pyplot as plt

name = ''
folder = ''
folder_path = f'{folder}/{name}_rasp'

file_list = os.listdir(folder_path)
file_list = sorted([file for file in file_list if file.endswith('.npy')], key = lambda x: int(os.path.splitext(x)[0]))
for file_name in file_list:
    vector = np.load(os.path.join(folder_path, file_name))
    plt.figure(figsize = (8, 6))
    plt.hist(vector, bins = 50, alpha = 0.7)
    plt.title('Distribution of feature vectors')
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.grid(False)
    chart_name = os.path.splitext(file_name)[0] + '.png'
    plt.savefig(os.path.join(folder_path, chart_name))
    plt.close()
    print(f'Plotted and saved: {chart_name}')