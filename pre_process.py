from PIL import Image
import numpy as np
import os


def conver_to_npy(directory, save_name):
    print("doing", save_name)
    my_array = []
    files = os.listdir(directory)
    print('total files', len(files))
    for file in files:
        if file.endswith('.bmp'):
            im = Image.open(directory + file).convert('I')
            # print(im.format, im.size, im.mode)
            # print(file)
            if im.size[0]*im.size[1] == 784:
                p = np.array(im)
                p = 255 - p
                new = p.reshape(784,)
                my_array.append(new)
    final_array = np.array(my_array)
    np.save('data/'+save_name, final_array)


conver_to_npy('data/Egg/', 'egg')
conver_to_npy('data/Question/', 'question')
conver_to_npy('data/Mickey/', 'mickey')
conver_to_npy('data/SadFace/', 'sad')
