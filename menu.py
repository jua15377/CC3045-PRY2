import os,sys
import shutil
import time
from PIL import Image
import numpy as np


count_files = 0
def read_move(file_name):
    global count_files
    print('Moving file to: analized')
    src = os.path.join('load', file_name)
    new_file_name = file_name.replace('.bmp', '')
    dst = os.path.join('analized', new_file_name+str(count_files) + '.bmp')
    try:
        shutil.move(src, dst)
    except Exception:
        pass


# cleaning directories
def cleaning_stuff(directory='analized'):
    '''
    clean directories
    :param directory: the directory to be cleaned
    :return:
    '''
    print('cleaning stuff on: ', directory)
    for file in os.listdir(directory):
        try:
            os.remove(os.path.join(directory, file))
        except Exception:
            pass

# pruebas
# a = np.load('data/circle.npy')
# b = a[0].reshape(28,28)
# i1 = Image.fromarray(b)
# i1.show()
# i2 = Image.fromarray(a[1000].reshape(28,28))
# i2.show()
# i3 = Image.fromarray(a[2].reshape(28,28))
# i3.show()


# cleaning_stuff()
def predict():
    '''
    Listens for file on the /load directory
    Loads and transform the image an then moves /analized
    '''
    global count_files

    print('running, waiting to get a file')
    while True:
        time.sleep(1)
        files = os.listdir('load')

        for file in files:
            if file.endswith('.bmp'):
                print('new BMP found!')
                im = Image.open('load/' + file).convert('I')
                # print(im.format, im.size, im.mode)
                p = np.array(im)
                p = 255 - p
                img_2 = Image.fromarray(p)
                img_2.show('after analize')
                print(p)
                count_files += 1
                read_move(files[0])


def train():
    '''
    reads datasets on /data and train the neuronal network
    '''
    pass


if __name__ == "__main__":
    option = sys.argv[1] if len(sys.argv) > 1 else ""
    if option == "--train" or option == '-t':
        train()
    elif option == "--predict" or option == '-p':
        predict()
    elif option == "--clean" or option == '-c':
        cleaning_stuff()

    else:
        print("Usage: python3 menu.py <option>")
        print("\nOption can be one of:\n")
        print("-t --train: ", train.__doc__)
        print("-p --predict: ", predict.__doc__)
        print("-c --clean: ", cleaning_stuff.__doc__)
