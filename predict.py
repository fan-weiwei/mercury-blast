import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

import time

from keras.models import model_from_json
import numpy as np  # linear algebra
import pandas as pd
from tqdm import tqdm
import preprocessor
import cv2
from model import *

from config import *
from record import *


def run():

    model_name = 'super128'
    model_folder = models_path + '/' + model_name
    model_weights_path = model_folder + '/model.h5'
    model_predictions_path = model_folder + '/predictions.pickle'


    print('*** Importing Images ***')

    x_test = []
    file_names = preprocessor.test_file_names()

    for name in tqdm(file_names):
        #print(spectral64_path + '/test/{}.jpg'.format(name))
        #continue
        img = cv2.imread(visible128_path + '/test/{}.jpg'.format(name))
        x_test.append(img)

    #return
    x_data = np.array(x_test, np.float16) / 255.

    print(x_data.shape)

    print("*** Loading Model ***")
    # load json and create model
    json_file = open(models_path + '/model.json', 'r')
    model_json = json_file.read()
    json_file.close()
    model = model_from_json(model_json)

    model.load_weights(models_path + '/model.h5')

    print("*** Compiling Model ***")
    model.compile(loss='binary_crossentropy',
                  # We NEED binary here, since categorical_crossentropy l1 norms the output before calculating loss.
                  optimizer='adam',
                  metrics=['accuracy'])

    print("*** Generating Predictions ***")
    #scores = model.evaluate(x_train, y_train, verbose=0)
    #print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

    predictions = model.predict(x_data)

    thresholds = current_model.load_thresholds(models_path + '/threshold.pickle')

    predictions_labels = []
    for prediction in predictions:
        labels = [inv_mapping[i] for i, value in enumerate(prediction) if value > thresholds[i]]
        predictions_labels.append(labels)

    print(*predictions_labels[:10], sep='\n')

    to_remove = []

    tags_list = [None] * len(predictions_labels)
    for i, tags in enumerate(predictions_labels):
        filtered = list(set(tags) - set(to_remove))
        tags_list[i] = ' '.join(map(str, filtered))

    final_data = [[filename.split(".")[0], tags] for filename, tags in zip(file_names, tags_list)]
    final_df = pd.DataFrame(final_data, columns=['image_name', 'tags'])
    final_df.head()

    timestr = time.strftime("%Y%m%d_%H%M%S")

    final_df.to_csv(submission_path + '/submission_file_{}.csv'.format(timestr), index=False)