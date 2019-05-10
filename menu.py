import os,sys
import shutil
import time
from PIL import Image
import numpy as np
from network import *
from sklearn.model_selection import train_test_split
import warnings
import heapq

warnings.filterwarnings("ignore")

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


def get_anwer(result):
    # print('result', result)
    the_max, all_outputs = result
    top_3 = heapq.nlargest(3, range(len(all_outputs)), all_outputs.take)

    for answer in top_3:
        for name, number in RESULT_DICT.items():
            if number == answer:
                print('the prediction is {} with {}%'.format(name[:-4], all_outputs[answer]*100))


# cleaning_stuff()
def predict():
    '''
    Listens for file on the /load directory
    Loads and transform the image an then moves /analized
    '''
    global count_files
    b = np.load('memory/biases_3.npy', allow_pickle=True)
    w = np.load('memory/weights_3.npy', allow_pickle=True)
    net = Network([784, 100, 16, 10])
    net.load_training(w, b)
    # house = np.load('data/circle.npy', allow_pickle=True)
    # for element in house:
    #     print(net.predict(element.reshape(784,1)))


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
                # img_2 = Image.fromarray(p)
                # img_2.show('after analize')
                p = p.reshape(784,1)
                get_anwer(net.predict(p))
                count_files += 1
                read_move(files[0])
                print('--------------------------------------------------------')


def train():
    '''
    reads datasets on /data and train the neuronal network
    '''
    print('loading data')
    all_data = np.load('data/all_data.npy', allow_pickle=True)
    train, test = train_test_split(all_data, test_size=0.2, random_state=3010)
    # setup network
    net = Network([784, 64, 10])
    # training
    net.minibach_gradient_decent(train, 20, 10, 2.0)
    net.save_training('6')
    # 2 [784,100,16,10]
    # 3 [784,100,50,16,10]lr = 3.0
    # 4 [784,100,10]
    # 5 [784,16,16,10] lr=2.0
    # 6 [784,64,10] lr¿ 2.0

def cross_validation():
    '''
    do cross validation to the network
    :return:
    '''
    # ***load data***
    all_data = np.load('data/all_data.npy', allow_pickle=True)
    # ***split data***
    # alternative
    # numpy.split
    # numpy.random.shuffle(x)
    print('loading data')
    train, test = train_test_split(all_data, test_size=0.3, random_state=3010)
    validation, final_test = np.split(test, 2)
    # setup network
    print('setting up network')
    b = np.load('memory/biases_3.npy', allow_pickle=True)
    w = np.load('memory/weights_3.npy', allow_pickle=True)
    # 2 [784,100,16,10]
    # 3 [784,100,50,16,10]
    # 4 [784,100,10]
    # 5 [784,16,16,10]
    # 6 [784,64,10] 50
    net = Network([784, 100, 50, 16, 10])
    net.load_training(w, b)
    # validation
    val_hits = net.test_performance(validation)
    # test
    test_hits = net.test_performance(final_test)
    # show
    print('train data:', len(train))
    print('validation data: {} | Total hits: {} | performance = {}%'.format(len(validation), val_hits, val_hits/len(validation)*100))
    print('test data: {} | Total hits: {} | performance = {}%'.format(len(final_test), test_hits, test_hits/len(final_test)*100))



if __name__ == "__main__":
    option = sys.argv[1] if len(sys.argv) > 1 else ""
    if option == "--train" or option == '-t':
        train()
    elif option == "--predict" or option == '-p':
        predict()
    elif option == "--cross_val" or option == '-cv':
        cross_validation()
    elif option == "--clean" or option == '-c':
        cleaning_stuff()



    else:
        print("Usage: python3 menu.py <option>")
        print("\nOption can be one of:\n")
        print("-t --train: ", train.__doc__)
        print("-p --predict: ", predict.__doc__)
        print("-c --clean: ", cleaning_stuff.__doc__)
        print("-cv --clean: ", cross_validation().__doc__)
