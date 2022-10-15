import os
import cv2
import pickle
import numpy as np
from numpy.core.records import array
from sklearn.utils import shuffle
import tensorflow as tf
from shutil import copyfile
from os import path, makedirs
from tensorflow.keras import layers
import BalancedDataGenerator
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array



class DatasetGeneratorFlowers():

    def __init__(self, root_dir, augmentation=False, balanced_batch=False, percentage=0.8, IMG_SIZE=600):
        self.root_dir = root_dir
        self.train_dir = root_dir + '/train'
        self.validation_dir = root_dir + '/validation/'
        self.augmentation = augmentation
        self.balanced_batch = balanced_batch
        self.percentage = percentage
        self.train_len = 0
        self.validation_len = 0
        self.IMG_SIZE = IMG_SIZE
        self.CATEGORIES = ['daisy', 'dandelion']
        self.class_indices = dict(
            zip(self.CATEGORIES, range(len(self.CATEGORIES))))
        if not path.exists(self.train_dir) or not path.exists(self.validation_dir):
            self.build_set()
            print('Loading Train File and Dataset  ..........')
            self.X_Train_Daisy, self.Y_Train_Daisy, self.X_Train_Dandelion, self.Y_Train_Dandelion = self.writeDataset(
                isTrain=True)
            self.train_len = len(self.Y_Train_Daisy) + \
                len(self.Y_Train_Dandelion)
            print('Found ', self.train_len, ' images')
            print('Loading Train File and Dataset  ..........')
            self.X_Validation_Daisy, self.Y_Validation_Daisy, self.X_Validation_Dandelion, self.Y_Validation_Dandelion = self.writeDataset(
                isTrain=False)
            self.validation_len = len(self.Y_Train_Daisy) + \
                len(self.Y_Train_Dandelion)
            print('Found ', self.validation_len, ' images')
        else:
            try:
                # Read the Data from Pickle Object
                self.X_Train_Daisy = pickle.load(
                    open('Train_X_Data_Daisy', 'rb'))
                self.X_Train_Dandelion = pickle.load(
                    open('Train_X_Data_Dandelion', 'rb'))

                self.Y_Train_Daisy = pickle.load(
                    open('Train_Y_Data_Daisy', 'rb'))
                self.Y_Train_Dandelion = pickle.load(
                    open('Train_Y_Data_Dandelion', 'rb'))

                self.train_len = len(self.Y_Train_Daisy) + \
                    len(self.Y_Train_Dandelion)
            except:
                print('Could not Found Pickle File ')
                print('Loading Train File and Dataset  ..........')
                self.X_Train_Daisy, self.Y_Train_Daisy, self.X_Train_Dandelion, self.Y_Train_Dandelion = self.writeDataset(
                    isTrain=True)
                self.train_len = len(self.Y_Train_Daisy) + \
                    len(self.Y_Train_Dandelion)
                print('Found ', self.train_len, ' images')
            try:
                # Read the Data from Pickle Object
                self.X_Validation_Daisy = pickle.load(
                    open('Validation_X_Data_Daisy', 'rb'))
                self.X_Validation_Dandelion = pickle.load(
                    open('Validation_X_Data_Dandelion', 'rb'))

                self.Y_Validation_Daisy = pickle.load(
                    open('Validation_Y_Data_Daisy', 'rb'))
                self.Y_Validation_Dandelion = pickle.load(
                    open('Validation_Y_Data_Dandelion', 'rb'))

                self.validation_len = len(self.Y_Validation_Daisy) + \
                    len(self.Y_Validation_Dandelion)
            except:
                print('Could not Found Pickle File ')
                print('Loading Train File and Dataset  ..........')
                self.X_Validation_Daisy, self.Y_Validation_Daisy, self.X_Validation_Dandelion, self.Y_Validation_Dandelion = self.writeDataset(
                    isTrain=False)
                self.validation_len = len(self.Y_Train_Daisy) + \
                    len(self.Y_Train_Dandelion)
                print('Found ', self.validation_len, ' images')

    def convert_to_numpy(self, x_data, y_data):
        x_data = np.asarray(x_data, dtype='float32')
        y_data = np.asarray(y_data)
        x_data = x_data.reshape(-1, self.IMG_SIZE, self.IMG_SIZE, 3)
        y_data = to_categorical(y_data)
        return x_data, y_data

    def process_image(self, path, category):
        """
            Return Numpy array of image
            :return: X_Data, Y_Data
        """
        try:
            image_data = []
            x_data = []
            y_data = []

            train_folder_path = os.path.join(path, category)  # Folder Path
            class_index = self.CATEGORIES.index(category)
            # This will iterate in the Folder
            for img in os.listdir(train_folder_path):
                # image Path
                new_path = os.path.join(train_folder_path, img)

                try:        # if any image is corrupted
                    # Read Image as numbers
                    _img = load_img(path=new_path,
                           color_mode='rgb',
                           target_size=(self.IMG_SIZE,self.IMG_SIZE))
                    image_data.append([img_to_array(_img), class_index])
                except:
                    pass

            # Iterate over the Data
            for features, labels in image_data:
                x_data.append(features)        # Get the X_Data
                y_data.append(labels)          # get the label
            image_data.clear()

            return x_data, y_data
        except:
            print("Failed to run Function Process Image ")

    def writeDataset(self, isTrain=True):
        """
        :return: None Creates a Pickle Object of DataSet
        """
        # Call the Function and Get the Data
        if(isTrain):
            X_Data_Daisy, Y_Data_Daisy = self.process_image(
                self.train_dir, 'daisy')
            X_Data_Dandelion, Y_Data_Dandelion = self.process_image(
                self.train_dir, 'dandelion')
            _type = 'Train_'
        else:
            X_Data_Daisy, Y_Data_Daisy = self.process_image(
                self.validation_dir, 'daisy')
            X_Data_Dandelion, Y_Data_Dandelion = self.process_image(
                self.validation_dir, 'dandelion')
            _type = 'Validation_'

        # Write the Entire Data into a Pickle File
        pickle_out = open(_type + 'X_Data_Daisy', 'wb')
        pickle.dump(X_Data_Daisy, pickle_out)
        pickle_out.close()

        pickle_out = open(_type + 'X_Data_Dandelion', 'wb')
        pickle.dump(X_Data_Dandelion, pickle_out)
        pickle_out.close()

        # Write the Y Label Data
        pickle_out = open(_type + 'Y_Data_Daisy', 'wb')
        pickle.dump(Y_Data_Daisy, pickle_out)
        pickle_out.close()

        pickle_out = open(_type + 'Y_Data_Dandelion', 'wb')
        pickle.dump(Y_Data_Dandelion, pickle_out)
        pickle_out.close()

        print("Pickled Image Successfully ")
        return X_Data_Daisy, Y_Data_Daisy, X_Data_Dandelion, Y_Data_Dandelion

    def build_set(self):
        for categories in self.CATEGORIES:
            makedirs(self.train_dir + '/' + categories)
            makedirs(self.validation_dir + '/' + categories)

        train_len = 0
        validation_len = 0

        for categories in self.CATEGORIES:
            print('separating ' + categories + ' files')
            allFileNames = os.listdir(self.root_dir + '/' + categories)
            np.random.shuffle(allFileNames)
            split_in = int(len(allFileNames) * self.percentage)

            for filename in allFileNames[0:split_in]:
                copyfile(self.root_dir + '/' + categories+'/' + filename,
                         self.train_dir + '/' + categories + '/' + filename,)
                train_len += 1
            for filename in allFileNames[split_in:]:
                copyfile(self.root_dir + '/' + categories+'/' + filename,
                         self.validation_dir + '/' + categories + '/' + filename,)
                train_len += 1

        self.train_len = train_len
        self.validation_len = validation_len

    def create_generators(self, batch_size):

        # this is the augmentation configuration we will use for testing:
        # only rescaling
        datagen = None
        if(self.augmentation):
            print('---------augmentation process------------------------------------')
            data_augmentation = tf.keras.Sequential([
                layers.experimental.preprocessing.RandomContrast(0.15),
                layers.experimental.preprocessing.RandomFlip(
                    "horizontal_and_vertical"),
                layers.experimental.preprocessing.RandomRotation(0.2),
                layers.experimental.preprocessing.RandomTranslation(0.2, 0.2),
                layers.experimental.preprocessing.RandomZoom(0.2)
            ])
            self.X_Train_Daisy = shuffle(self.X_Train_Daisy)
            class_index = self.CATEGORIES.index('daisy')
            for i in range(0, len(self.X_Train_Daisy)):
                _img = data_augmentation(
                    tf.expand_dims(self.X_Train_Daisy[i], axis=0))
                self.X_Train_Daisy.append(tf.squeeze(_img).numpy())
                self.Y_Train_Daisy.append(class_index)

            self.train_len *= 2
            print('--------- augmentation process complete ---------------------------')

        self.X_Train = self.X_Train_Dandelion
        self.Y_Train = self.Y_Train_Dandelion
        for i in range(0, len(self.X_Train_Daisy)):
            self.X_Train.append(self.X_Train_Daisy[i])
            self.Y_Train.append(self.Y_Train_Daisy[i])
        self.X_Train, self.Y_Train = self.convert_to_numpy(
            self.X_Train, self.Y_Train)
        if(self.balanced_batch):
            print('---------balanced batch process-----------------------------------')
            datagen = ImageDataGenerator()
            train_generator = BalancedDataGenerator.BalancedDataGenerator(
                self.X_Train,
                self.Y_Train,
                datagen,
                batch_size=batch_size,
                classes=self.CATEGORIES)
            steps_train = train_generator.steps_per_epoch
            y_gen = [train_generator.__getitem__(
                0)[1] for i in range(steps_train)]
            print(np.unique(y_gen, return_counts=True))
            print(train_generator.class_indices)
            print('--------- balanced batch process complete ---------------------------')
        else:
            train_datagen = ImageDataGenerator()
            train_generator = train_datagen.flow(
                self.X_Train,
                self.Y_Train,
                batch_size=batch_size,)
            steps_train = self.train_len//batch_size

        self.X_Validation = self.X_Validation_Dandelion
        self.Y_Validation = self.Y_Validation_Dandelion
        for i in range(0, len(self.X_Validation_Daisy)):
            self.X_Validation.append(self.X_Validation_Daisy[i])
            self.Y_Validation.append(self.Y_Validation_Daisy[i])
        self.X_Validation, self.Y_Validation = self.convert_to_numpy(
            self.X_Validation, self.Y_Validation)
        test_datagen = ImageDataGenerator()
        validation_generator = test_datagen.flow(
            self.X_Validation,
            self.Y_Validation,
            batch_size=batch_size,)
        steps_validation = self.validation_len//batch_size

        return train_generator, validation_generator, steps_train, steps_validation
