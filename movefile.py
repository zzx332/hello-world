import shutil
import os
data_dir = 'D:/data/birddata/CUB_200_2011/images/'  # type: str
x = 'D:/data/birddata/CUB_200_2011/data/'
n = 0
for train_class in os.listdir(data_dir):
    n = n + 1
    p = 0
    if n > 101:
        for pic in os.listdir(data_dir + train_class):
            p = p + 1
            if p < 6:
                shutil.move(data_dir + train_class + '/' + pic, x + train_class)






