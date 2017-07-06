from abc import ABC, abstractmethod
import preprocessor
import cv2
import numpy as np

from tqdm import tqdm
from config import *
from record import *
from keras.layers import *
from keras.models import *
import pickle


class Model(ABC):

    @abstractmethod
    def training_images(self):
        pass

    @staticmethod
    def load_annotations(data):

        annotations = []

        for item in data:
            targets = np.zeros(17)
            for t in item.annotations:
                targets[mapping[t]] = 1

            annotations.append(targets)

        return np.array(annotations, np.uint8)

class Super128(Model):

    def __init__(self):
        data = preprocessor.read_data(train_csv_path)

        self.training_data = data[:30000]
        self.validation_data = data[30000:]
        self.compile_model()
        self.thresholds = []

    def load_images(self, data):

        print('*** loading images ***')
        images = []

        for item in tqdm(data):
            image = cv2.imread(visible128_path + '/train/{}.jpg'.format(item.name))
            images.append(image)

        return np.array(images, np.float16) / 255.



    def compile_model(self):

        model = Sequential()

        model.add(BatchNormalization(input_shape=(128, 128, 3)))
        model.add(Conv2D(28, (3, 1), padding='same', activation='relu'))
        model.add(Conv2D(28, (1, 3), activation='relu'))
        model.add(MaxPooling2D((2, 2), strides=(2, 2)))

        model.add(BatchNormalization())
        model.add(Conv2D(48, (3, 2), padding='same', activation='relu'))
        model.add(Conv2D(48, (2, 3), activation='relu'))
        model.add(MaxPooling2D((2, 2), strides=(2, 2)))
        model.add(Dropout(0.30))

        model.add(BatchNormalization())
        model.add(Conv2D(128, (3, 2), padding='same', activation='relu'))
        model.add(Conv2D(128, (2, 3), activation='relu'))
        model.add(MaxPooling2D((2, 2), strides=(2, 2)))
        model.add(Dropout(0.70))

        model.add(BatchNormalization())
        model.add(Conv2D(256, (3, 2), padding='same', activation='relu'))
        model.add(Conv2D(256, (2, 3), activation='relu'))
        model.add(MaxPooling2D((2, 2), strides=(2, 2)))
        model.add(Dropout(0.90))

        model.add(Flatten())
        model.add(Dense(512, activation='relu', use_bias=False))
        model.add(BatchNormalization())
        model.add(Dense(17, activation='sigmoid'))

        opt = optimizers.Adam(lr=0.001)
        print("*** Compiling Model ***")
        model.compile(loss='binary_crossentropy',
                      # We NEED binary here, since categorical_crossentropy l1 norms the output before calculating loss.
                      optimizer=opt,
                      metrics=['accuracy'])

        self.model = model

    def load_weights(self, string):
        self.model.load_weights(string)

    def save_weights(self, string):
        self.model.save_weights(string)

    def save_thresholds(self, string):
        with open(string, 'wb') as handle:
            pickle.dump(self.thresholds, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def load_thresholds(self, string):
        with open(string, 'rb') as handle:
            unserialized_data = pickle.load(handle)
        return unserialized_data

    def calc_thresholds(self):


        print('thresholds')

        pass


    #TODO Make image set enum?
    def training_images(self):
        return self.load_images(self.training_data)

    def validation_images(self):
        return self.load_images(self.validation_data)

    def training_some_hot(self):
        return self.load_annotations(self.training_data)

    def validation_some_hot(self):
        return self.load_annotations(self.validation_data)

    def training_file_names(self):
        return [x.name for x in self.training_data]

    def validation_file_names(self):
        return [x.name for x in self.validation_data]




current_model = Super128()

