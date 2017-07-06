import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

from keras.models import model_from_json
import numpy as np  # linear algebra
import pandas as pd
from tqdm import tqdm
import preprocessor
import models

from config import *

def run():
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

    x_train = []
    y_train = []

    data = preprocessor.read_data(train_csv_path)[30000:]

    file_names = [x.name for x in data]

    print('*** Importing images ***')

    for item in tqdm(data):
        x_train.append(models.AnnotatedRecord.visible128(item))
        y_train.append(models.AnnotatedRecord.some_hot(item))

    x_data = np.array(x_train, np.float16) / 255.
    y_data = np.array(y_train, np.uint8)

    print(x_data.shape)
    print(y_data.shape)


    print("*** Loading Model ***")
    # load json and create model
    json_file = open(models_path + '/alt128/model.json', 'r')
    model_json = json_file.read()
    json_file.close()
    model = model_from_json(model_json)

    model.load_weights(models_path + '/alt128/model.h5')

    print("*** Compiling Model ***")
    model.compile(loss='binary_crossentropy',
                  # We NEED binary here, since categorical_crossentropy l1 norms the output before calculating loss.
                  optimizer='adam',
                  metrics=['accuracy'])

    predictions = model.predict(x_data)

    thresholds = [0.2] * len(models.inv_mapping)

    predictions_labels = []
    for prediction in predictions:
        labels = [models.inv_mapping[i] for i, value in enumerate(prediction) if value > thresholds[i]]
        predictions_labels.append(labels)

    #print(*predictions_labels[:10], sep='\n')

    y_labels = []
    for record in y_data:
        labels = [models.inv_mapping[i] for i, value in enumerate(record) if value == 1]
        y_labels.append(labels)
    #print(*y_labels[:10], sep='\n')

    predict_data = []
    for prediction in predictions:
        predict_data.append(list(map(lambda x: int(x > .20), prediction)))

    #print(*predict_data[:5], sep='\n')
    #print(*y_data[:5], sep='\n')

    total_num = y_data.__len__()
    print("Total number of items: {}".format(total_num))

    for index in range(17):

        print("\nFeature: ", models.inv_mapping[index], "\n")

        total_features = sum([x[index]==1 for x in y_data])
        print("Total: ", total_features)

        true_positives = sum([x[index]==1 and y[index]==1 for x, y in zip(y_data, predict_data)])
        print("True Positives: ", true_positives)

        false_negatives = sum([x[index]==1 and y[index]==0 for x, y in zip(y_data, predict_data)])
        print("False Negatives: ", false_negatives)

        false_positives = sum([x[index]==0 and y[index]==1 for x, y in zip(y_data, predict_data)])
        print("False Positives: ", false_positives)

        accuracy = true_positives / (true_positives + false_positives)
        print("Accuracy ", accuracy)

        recall = true_positives / (true_positives + false_negatives)
        print("Recall ", recall)

    tags_list = [None] * len(predictions_labels)
    for i, tags in enumerate(predictions_labels):
        tags_list[i] = ' '.join(map(str, tags))

    final_data = [[filename.split(".")[0], tags] for filename, tags in zip(file_names, tags_list)]
    final_df = pd.DataFrame(final_data, columns=['image_name', 'tags'])
    final_df.head()

    final_df.to_csv(diag_path + '/alt128.csv', index=False)