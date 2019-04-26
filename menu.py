import os
import shutil
import time

print('running, wating to get a file')

count_files = 0
def read_move(file_name):
    print('Moving file to: analized')
    src = os.path.join('load', file_name)
    new_file_name = file_name.replace('.bmp', '')
    dst = os.path.join('analized', new_file_name+str(count_files) + '.bmp')
    try:
        shutil.move(src, dst)
    except Exception:
        pass


# cleaning directories
def cleaning_stuff(directory = 'analized'):
    print('cleaning stuff on: ', directory)
    for file in os.listdir(directory):
        try:
            os.remove(os.path.join(directory, file))
        except Exception:
            pass

# cleaning_stuff()
while True:
    time.sleep(1)
    files = os.listdir('load')

    for file in files:
        if file.endswith('.bmp'):
            print('new BMP found!')
            count_files += 1
            read_move(files[0])
