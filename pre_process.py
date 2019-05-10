from PIL import Image
import numpy as np
import os

RESULT_DICT = {
    'circle.npy': 0,
    'square.npy': 1,
    'triangle.npy': 2,
    'egg.npy': 3,
    'tree.npy': 4,
    'house.npy': 5,
    'face.npy': 6,
    'sad.npy': 7,
    'question.npy': 8,
    'mickey.npy': 9
}

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


def vectorized_result(j):
    e = np.zeros((10, 1))
    e[j] = 1.0
    return e


def merge_data():
    my_array = []
    files = os.listdir('data/')
    print('total files', len(files))
    for file in files:
        if file.endswith('.npy'):
            print(file)
            npy = np.load(os.path.join('data/', file))
            for element in npy:
                my_array.append((element.reshape(784, 1), vectorized_result(RESULT_DICT[file])))
    print('finish')
    final_array = np.array(my_array)
    np.save('data/'+'all_data', my_array)



# use this when you need o recreate the data set from the sample image
# conver_to_npy('data/Egg/', 'egg')
# conver_to_npy('data/Question/', 'question')
# conver_to_npy('data/Mickey/', 'mickey')
# conver_to_npy('data/SadFace/', 'sad')
merge_data()