import sys
import os
import numpy as np
from numpy.core.records import array
from sklearn.utils import shuffle
import tensorflow as tf
from shutil import copyfile
from os import path, makedirs
from tensorflow.keras import layers

from tensorflow.keras.preprocessing.image import ImageDataGenerator

if __name__ == '__main__':
    data_dir = sys.argv[1]
    train_dir = data_dir + '/train-set'
    test_dir = data_dir + '/test-set'
    CATEGORIES = ['daisy', 'dandelion']
    for categories in CATEGORIES:
        allFileNames = os.listdir(data_dir + '/' + categories)
        np.random.shuffle(allFileNames)
        split_in = int(len(allFileNames) * 0.8)

        for filename in allFileNames[0:split_in]:
            copyfile(data_dir + '/' + categories+'/' + filename,
                     train_dir + '/' + categories + '/' + filename)

        for filename in allFileNames[split_in:]:
            copyfile(data_dir + '/' + categories+'/' + filename,
                     test_dir + '/' + categories + '/' + filename)