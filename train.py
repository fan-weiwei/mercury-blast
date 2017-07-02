import models
import numpy as np  # linear algebra
from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense, BatchNormalization
from keras.models import Sequential, model_from_json
from tqdm import tqdm

import preprocessor
from config import *
import os

def run():

    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

    x_train = []
    x_test = []
    y_train = []

    data = preprocessor.read_data(train_csv_path)

    print('*** Importing images ***')

    for item in tqdm(data):
        x_train.append(models.AnnotatedRecord.spectral_image(item))
        y_train.append(models.AnnotatedRecord.some_hot(item))

    x_data = np.array(x_train, np.float16) / 255.
    y_data = np.array(y_train, np.uint8)

    print(x_data.shape)
    print(y_data.shape)

    split = 35000
    x_train = x_data[:split]
    x_valid = x_data[split:]
    y_train = y_data[:split]
    y_valid = y_data[split:]

    '''
    model = Sequential()
    model.add(BatchNormalization(input_shape=(64, 64, 3)))

    model.add(Conv2D(32, (3, 3), padding='same', activation='relu'))
    model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=2))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=2))
    model.add(Dropout(0.25))

    model.add(Conv2D(128, (3, 3), padding='same', activation='relu'))
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=2))
    model.add(Dropout(0.25))

    model.add(Conv2D(256, (3, 3), padding='same', activation='relu'))
    model.add(Conv2D(256, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=2))
    model.add(Dropout(0.25))

    model.add(Flatten())

    #model.add(Dense(64, bias=False))?
    model.add(Dense(512, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Dense(17, activation='sigmoid'))

    '''

    print("*** Loading Model ***")
    # load json and create model
    json_file = open(models_path + '/model.json', 'r')
    model_json = json_file.read()
    json_file.close()
    model = model_from_json(model_json)

    model.load_weights(models_path + '/model.h5')

    print("*** Saving Old Model ***")
    # Serialize to JSON
    model_json = model.to_json()
    with open(models_path + '/model-old.json', 'w') as json_file:
        json_file.write(model_json)

    model.save_weights(models_path + '/model-old.h5')



    print("*** Compiling Model ***")
    model.compile(loss='binary_crossentropy',
                  # We NEED binary here, since categorical_crossentropy l1 norms the output before calculating loss.
                  optimizer='adam',
                  metrics=['accuracy'])


    model.fit(x_train, y_train,
              batch_size=128,
              epochs=50,
              verbose=1,
              validation_data=(x_valid, y_valid))

    print("*** Evaluating Model ***")
    scores = model.evaluate(x_train, y_train, verbose=0)
    print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

    print("*** Saving Model ***")
    # Serialize to JSON
    model_json = model.to_json()
    with open(models_path + '/model.json', 'w') as json_file:
        json_file.write(model_json)

    model.save_weights(models_path + '/model.h5')
    print("Saved model to disk")



    #from sklearn.metrics import fbeta_score

    #p_valid = model.predict(x_valid, batch_size=128)
    #print(y_valid)
    #print(p_valid)
    #print(fbeta_score(y_valid, np.array(p_valid) > 0.2, beta=2, average='samples'))


