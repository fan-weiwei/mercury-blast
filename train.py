from keras import callbacks
from keras import optimizers
from keras.callbacks import ModelCheckpoint

import time
from keras.layers import *
from keras.models import *
from tensorflow.contrib.keras.api.keras.callbacks import Callback, EarlyStopping

import os

from model import *

def run():

    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

    x_data = current_model.training_images()
    y_data = current_model.training_some_hot()

    print(x_data.shape)
    print(y_data.shape)

    split = 30000
    x_train = x_data[:split]
    x_valid = x_data[split:]
    y_train = y_data[:split]
    y_valid = y_data[split:]

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

    '''
    print("*** Saving Model ***")
    # Serialize to JSON
    model_json = model.to_json()
    with open(models_path + '/model.json', 'w') as json_file:
        json_file.write(model_json)
  
    print("*** Loading Model ***")
     load json and create model
    json_file = open(models_path + '/model.json', 'r')
    model_json = json_file.read()
    json_file.close()
    model = model_from_json(model_json)
    '''
    #print("*** Loading Weights ***")
    #model.load_weights(models_path + '/model.h5')

    '''
    print("*** Saving Old Model ***")
    # Serialize to JSON
    model_json = model.to_json()
    with open(models_path + '/model-old.json', 'w') as json_file:
        json_file.write(model_json)

    '''
    #model.save_weights(models_path + '/model-old.h5')


    opt = optimizers.Adam(lr=0.001)
    print("*** Compiling Model ***")
    model.compile(loss='binary_crossentropy',
                  # We NEED binary here, since categorical_crossentropy l1 norms the output before calculating loss.
                  optimizer=opt,
                  metrics=['accuracy'])

    earlyStopping = EarlyStopping(monitor='val_loss', patience=5, verbose=0, mode='auto')

    timestr = time.strftime("%Y%m%d_%H%M%S")

    checkpointer = ModelCheckpoint(filepath=models_path + '/model.h5', verbose=0, save_best_only=True)
    tensorBoard = callbacks.TensorBoard(log_dir="logs/{}".format(timestr), histogram_freq=0, write_graph=True, write_images=True)

    print("*** Training ***")
    model.fit(x_train, y_train,
              batch_size=128,
              epochs= 1,
              verbose=1,
              validation_data=(x_valid, y_valid),
              callbacks=[tensorBoard, checkpointer, earlyStopping])



    #model.save_weights(models_path + '/model.h5')


    #from sklearn.metrics import fbeta_score

    #p_valid = model.predict(x_valid, batch_size=128)
    #print(y_valid)
    #print(p_valid)
    #print(fbeta_score(y_valid, np.array(p_valid) > 0.2, beta=2, average='samples'))


