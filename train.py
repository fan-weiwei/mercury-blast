from keras import callbacks
from keras import optimizers
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau

import time

from keras.preprocessing.image import ImageDataGenerator
from tensorflow.contrib.keras.api.keras.callbacks import Callback, EarlyStopping

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from model import *

def run():

    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

    x_valid = current_model.split.validation_images()
    y_valid = current_model.split.validation_some_hot()

    #x_train = current_model.split.training_images()
    #y_train = current_model.split.training_some_hot()


    print("*** Training ***")

    current_model.load_weights(models_path + '/model-old.h5')
    current_model.save_weights(models_path + '/model-old.h5')

    early_stopping = EarlyStopping(monitor='val_loss', patience=12, verbose=0, mode='auto')
    timestr = time.strftime("%Y%m%d_%H%M%S")
    checkpointer = ModelCheckpoint(filepath=models_path + '/checkpoint-model.h5', verbose=0, save_best_only=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2,
                                  patience=5, min_lr=0.0001)
    #tensor_board = callbacks.TensorBoard(log_dir="logs/{}".format(timestr), histogram_freq=0, write_graph=True, write_images=True)

    train_datagen = ImageDataGenerator(
        fill_mode="reflect",
        horizontal_flip=True,
        zoom_range=0.30,
        width_shift_range=0.30,
        height_shift_range=0.30,
        rotation_range=360)

    #for n in range(100):

    x_train = current_model.split.training_images()
    y_train = current_model.split.training_some_hot()

    current_model.model.fit_generator(train_datagen.flow(x_train, y_train, batch_size=128),
              steps_per_epoch=len(x_train) / 128,
              epochs= 2000,
              verbose=1,
              validation_data=(x_valid, y_valid),
              callbacks=[checkpointer])



    current_model.save_weights(models_path + '/model.h5')


        #current_model.save_weights(models_path + '/model.h5')





