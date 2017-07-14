from abc import ABC, abstractmethod
from preprocessor import read_data
import cv2
import numpy as np

from tqdm import tqdm
from config import *
from record import *
from keras.layers import *
from keras.models import *
import pickle

class Ab_Model(ABC):

    #@abstractmethod
    #def training_images(self):
    #    pass

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


        # TODO Make image set enum?



class default_split:

    def __init__(self, training_path):
        data = read_data(train_csv_path)
        self.training_data = data[:30000]
        self.validation_data = data[30000:]

        self.training_path = training_path

    def load_images(self, data):

        print('*** loading images ***')
        images = []

        for item in tqdm(data):
            image = cv2.imread(self.training_path.format(item.name))
            images.append(image)

        return np.array(images, np.float16) / 255.

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

    @staticmethod
    def load_annotations(data):

        annotations = []

        for item in data:
            targets = np.zeros(17)
            for t in item.annotations:
                targets[mapping[t]] = 1

            annotations.append(targets)

        return np.array(annotations, np.uint8)

class batch_split:

    def __init__(self, training_path):
        data = read_data(train_csv_path)
        self.training_data = data[:30000]
        self.validation_data = data[33000:]
        self.training_path = training_path
        self.batch_data = self.randomize_batch()


    def load_images(self, data):

        print('*** loading images ***')
        images = []

        for item in tqdm(data):
            image = cv2.imread(self.training_path.format(item.name))
            images.append(image)

        return np.array(images, np.float16) / 255.

    def randomize_batch(self):
        return np.random.choice(self.training_data, 16000, replace=False)

    def training_images(self):
        return self.load_images(self.batch_data)

    def validation_images(self):
        return self.load_images(self.validation_data)

    def training_some_hot(self):
        return self.load_annotations(self.batch_data)

    def validation_some_hot(self):
        return self.load_annotations(self.validation_data)

    def training_file_names(self):
        return [x.name for x in self.training_data]

    def validation_file_names(self):
        return [x.name for x in self.validation_data]

    @staticmethod
    def load_annotations(data):

        annotations = []

        for item in data:
            targets = np.zeros(17)
            for t in item.annotations:
                targets[mapping[t]] = 1

            annotations.append(targets)

        return np.array(annotations, np.uint8)

class Spectral128(Ab_Model):

    def __init__(self):
        self.split = batch_split(spectral128_path + '/train/{}.jpg')
        self.model = self.compile_model()
        self.thresholds = []

    @staticmethod
    def compile_model():

        model = Sequential()

        model.add(BatchNormalization(input_shape=(128, 128, 3)))
        model.add(Conv2D(28, (3, 1), padding='same', activation='relu'))
        model.add(Conv2D(28, (1, 3), activation='relu'))
        model.add(MaxPooling2D((2, 2), strides=(2, 2)))

        model.add(BatchNormalization())
        model.add(Conv2D(48, (3, 2), padding='same', activation='relu'))
        model.add(Conv2D(48, (2, 3), activation='relu'))
        model.add(MaxPooling2D((2, 2), strides=(2, 2)))
        model.add(Dropout(0.10))

        model.add(BatchNormalization())
        model.add(Conv2D(128, (3, 2), padding='same', activation='relu'))
        model.add(Conv2D(128, (2, 3), activation='relu'))
        model.add(MaxPooling2D((2, 2), strides=(2, 2)))
        model.add(Dropout(0.25))

        model.add(BatchNormalization())
        model.add(Conv2D(256, (3, 2), padding='same', activation='relu'))
        model.add(Conv2D(256, (2, 3), activation='relu'))
        model.add(MaxPooling2D((2, 2), strides=(2, 2)))
        model.add(Dropout(0.90))

        model.add(Flatten())
        model.add(Dense(512, activation='relu', use_bias=False))
        model.add(BatchNormalization())
        model.add(Dense(17, activation='sigmoid'))

        opt = optimizers.Nadam(lr=0.002)
        print("*** Compiling Model ***")
        model.compile(loss='binary_crossentropy',
                      # We NEED binary here, since categorirom keras import applicationscal_crossentropy l1 norms the output before calculating loss.
                      optimizer=opt,
                      metrics=['accuracy'])

        return model

class Super128(Ab_Model):

    def __init__(self):
        self.split = default_split(visible128_path + '/train/{}.jpg')
        self.model = self.compile_model()
        self.thresholds = []

    @staticmethod
    def compile_model():

        model = Sequential()

        model.add(BatchNormalization(input_shape=(128, 128, 3)))
        model.add(Conv2D(28, (3, 1), padding='same', activation='relu'))
        model.add(Conv2D(28, (1, 3), activation='relu'))
        model.add(MaxPooling2D((2, 2), strides=(2, 2)))

        model.add(BatchNormalization())
        model.add(Conv2D(48, (3, 2), padding='same', activation='relu'))
        model.add(Conv2D(48, (2, 3), activation='relu'))
        model.add(MaxPooling2D((2, 2), strides=(2, 2)))
        model.add(Dropout(0.10))

        model.add(BatchNormalization())
        model.add(Conv2D(128, (3, 2), padding='same', activation='relu'))
        model.add(Conv2D(128, (2, 3), activation='relu'))
        model.add(MaxPooling2D((2, 2), strides=(2, 2)))
        model.add(Dropout(0.25))

        model.add(BatchNormalization())
        model.add(Conv2D(256, (3, 2), padding='same', activation='relu'))
        model.add(Conv2D(256, (2, 3), activation='relu'))
        model.add(MaxPooling2D((2, 2), strides=(2, 2)))
        model.add(Dropout(0.90))

        model.add(Flatten())
        model.add(Dense(512, activation='relu', use_bias=False))
        model.add(BatchNormalization())
        model.add(Dense(17, activation='sigmoid'))

        opt = optimizers.Nadam(lr=0.002)
        #opt = optimizers.SGD(lr=0.005, decay=1e-6, momentum=0.9, nesterov=True)
        print("*** Compiling Model ***")
        model.compile(loss='binary_crossentropy',
                      # We NEED binary here, since categorical_crossentropy l1 norms the output before calculating loss.
                      optimizer=opt,
                      metrics=['accuracy'])

        return model

class Xin128(Ab_Model):

    def __init__(self):
        self.split = default_split(visible128_path + '/train/{}.jpg')
        self.model = self.compile_model()
        self.thresholds = []

    @staticmethod
    def compile_model():
        model = Sequential()

        model.add(BatchNormalization(input_shape=(128, 128, 3)))
        model.add(Conv2D(28, (3, 3), padding='same', activation='relu'))
        model.add(Conv2D(28, (3, 3), activation='relu'))
        model.add(MaxPooling2D((2, 2), strides=(2, 2)))

        model.add(BatchNormalization())
        model.add(Conv2D(48, (3, 3), padding='same', activation='relu'))
        model.add(Conv2D(48, (3, 3), activation='relu'))
        model.add(MaxPooling2D((2, 2), strides=(2, 2)))
        model.add(Dropout(0.50))

        model.add(BatchNormalization())
        model.add(Conv2D(128, (3, 3), padding='same', activation='relu'))
        model.add(Conv2D(128, (3, 3), activation='relu'))
        model.add(MaxPooling2D((2, 2), strides=(2, 2)))
        model.add(Dropout(0.70))

        model.add(BatchNormalization())
        model.add(Conv2D(128, (3, 3), padding='same', activation='relu'))
        model.add(Conv2D(128, (3, 3), activation='relu'))
        model.add(MaxPooling2D((2, 2), strides=(2, 2)))
        model.add(Dropout(0.70))

        model.add(BatchNormalization())
        model.add(Conv2D(256, (3, 3), padding='same', activation='relu'))
        model.add(Conv2D(256, (3, 3), activation='relu'))
        model.add(MaxPooling2D((2, 2), strides=(2, 2)))
        model.add(Dropout(0.70))

        model.add(Flatten())
        model.add(Dense(512, activation='relu', use_bias=False))
        model.add(BatchNormalization())
        model.add(Dense(512, activation='relu', use_bias=False))
        model.add(BatchNormalization())
        model.add(Dense(17, activation='sigmoid'))

        opt = optimizers.Nadam(lr=0.002*0.05)
        print("*** Compiling Model ***")
        model.compile(loss='binary_crossentropy',
                      # We NEED binary here, since categorirom keras import applicationscal_crossentropy l1 norms the output before calculating loss.
                      optimizer=opt,
                      metrics=['accuracy'])

        return model

class Super224(Ab_Model):

    def __init__(self):
        self.split = batch_split(visible224_path + '/train/{}.jpg')
        self.model = self.compile_model()
        self.thresholds = []

    @staticmethod
    def compile_model():

        model = Sequential()

        model.add(BatchNormalization(input_shape=(224, 224, 3)))
        model.add(Conv2D(28, (3, 1), padding='same', activation='relu'))
        model.add(Conv2D(28, (1, 3), activation='relu'))
        model.add(MaxPooling2D((2, 2), strides=(2, 2)))

        model.add(BatchNormalization())
        model.add(Conv2D(48, (3, 2), padding='same', activation='relu'))
        model.add(Conv2D(48, (2, 3), activation='relu'))
        model.add(MaxPooling2D((2, 2), strides=(2, 2)))
        model.add(Dropout(0.10))

        model.add(BatchNormalization())
        model.add(Conv2D(128, (3, 2), padding='same', activation='relu'))
        model.add(Conv2D(128, (2, 3), activation='relu'))
        model.add(MaxPooling2D((2, 2), strides=(2, 2)))
        model.add(Dropout(0.25))

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
                      optimizer=opt,
                      metrics=['accuracy'])

        return model

from keras import applications
class VGG224(Ab_Model):

    def __init__(self):
        self.split = default_split(visible128_path + '/train/{}.jpg')
        self.model = self.compile_model()
        self.thresholds = []

    @staticmethod
    def compile_model():

        input_tensor = Input(shape=(128, 128, 3))

        base_model = applications.VGG16(weights='imagenet', include_top=False, input_tensor=input_tensor)
        print('Model loaded.')

        # build a classifier model to put on top of the convolutional model
        top_model = Sequential()
        top_model.add(Flatten(input_shape=base_model.output_shape[1:]))
        top_model.add(Dense(256, activation='relu'))
        top_model.add(Dropout(0.5))
        top_model.add(Dense(17, activation='sigmoid'))

        model = Model(inputs=base_model.input, outputs=top_model(base_model.output))

        # Freeze the layers which you don't want to train
        for layer in model.layers[:5]:
            layer.trainable = False

        opt = optimizers.Adam(lr=0.001)
        print("*** Compiling Model ***")
        model.compile(loss='binary_crossentropy',
                      optimizer=opt,
                         metrics=['accuracy'])

        return model

from keras import applications
class ResNet50(Ab_Model):

    def __init__(self):
        self.split = batch_split(visible200_path + '/train/{}.jpg')
        self.model = self.compile_model()
        self.thresholds = []

    @staticmethod
    def compile_model():

        input_tensor = Input(shape=(200, 200, 3))

        base_model = applications.ResNet50(weights='imagenet', include_top=False, input_tensor=input_tensor)
        print('Model loaded.')

        # Freeze the layers which you don't want to train
        for layer in base_model.layers[:46]:
            layer.trainable = False

        for layer in base_model.layers[48:50]:
            layer.trainable = False

        # build a classifier model to put on top of the convolutional model
        top_model = Sequential()

        top_model.add(Flatten(input_shape=base_model.output_shape[1:]))
        top_model.add(Dropout(0.90))
        top_model.add(Dense(512, activation='relu'))
        top_model.add(BatchNormalization())
        top_model.add(Dense(17, activation='sigmoid'))


        model = Model(inputs=base_model.input, outputs=top_model(base_model.output))

        #opt = optimizers.Nadam(lr=0.001)
        sgd = optimizers.SGD(lr=0.002, decay=1e-6, momentum=0.9, nesterov=True)
        print("*** Compiling Model ***")
        model.compile(loss='binary_crossentropy',
                      optimizer=sgd,
                      metrics=['accuracy'])

        return model


current_model = Xin128()

