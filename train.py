from keras import callbacks
from keras import optimizers
from keras.callbacks import ModelCheckpoint

import time

from tensorflow.contrib.keras.api.keras.callbacks import Callback, EarlyStopping

import os

from model import *

def run():

    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

    x_train = current_model.split.training_images()
    y_train = current_model.split.training_some_hot()

    x_valid = current_model.split.validation_images()
    y_valid = current_model.split.validation_some_hot()

    #model.save_weights(models_path + '/model-old.h5')
    #current_model.load_weights(models_path + '/model.h5')

    earlyStopping = EarlyStopping(monitor='val_loss', patience=5, verbose=0, mode='auto')

    timestr = time.strftime("%Y%m%d_%H%M%S")

    checkpointer = ModelCheckpoint(filepath=models_path + '/model.h5', verbose=0, save_best_only=True)
    tensorBoard = callbacks.TensorBoard(log_dir="logs/{}".format(timestr), histogram_freq=0, write_graph=True, write_images=True)

    print("*** Training ***")
    current_model.model.fit(x_train, y_train,
              batch_size=128,
              epochs= 1,
              verbose=1,
              validation_data=(x_valid, y_valid),
              callbacks=[tensorBoard, checkpointer, earlyStopping])

    #current_model.thresholds = [0.20] * 17
    #current_model.save_weights(models_path + '/model.h5')
    #current_model.calc_thresholds()
    #current_model.save_thresholds(models_path + '/threshold.pickle')
    #current_model.load_thresholds(models_path + '/threshold.pickle')


    #model.save_weights(models_path + '/model.h5')


    #from sklearn.metrics import fbeta_score

    #p_valid = model.predict(x_valid, batch_size=128)
    #print(y_valid)
    #print(p_valid)
    #print(fbeta_score(y_valid, np.array(p_valid) > 0.2, beta=2, average='samples'))


