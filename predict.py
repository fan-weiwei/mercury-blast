import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

import time

from keras.models import model_from_json
import numpy as np  # linear algebra
import pandas as pd
from tqdm import tqdm
import preprocessor
import models
import cv2

print('*** Importing Images ***')

x_test = []
file_names = preprocessor.test_file_names()

for name in tqdm(file_names):
    img = cv2.imread('../test-jpg-small/{}.jpg'.format(name))
    x_test.append(img)

x_data = np.array(x_test, np.float16) / 255.

print(x_data.shape)

print("*** Loading Model ***")
# load json and create model
json_file = open('model.json', 'r')
model_json = json_file.read()
json_file.close()
model = model_from_json(model_json)

model.load_weights("model.h5")

print("*** Compiling Model ***")
model.compile(loss='binary_crossentropy',
              # We NEED binary here, since categorical_crossentropy l1 norms the output before calculating loss.
              optimizer='adam',
              metrics=['accuracy'])

print("*** Generating Predictions ***")
#scores = model.evaluate(x_train, y_train, verbose=0)
#print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

predictions = model.predict(x_data)

thresholds = [0.2] * len(models.inv_mapping)

predictions_labels = []
for prediction in predictions:
    labels = [models.inv_mapping[i] for i, value in enumerate(prediction) if value > thresholds[i]]
    predictions_labels.append(labels)

print(*predictions_labels[:10], sep='\n')

tags_list = [None] * len(predictions_labels)
for i, tags in enumerate(predictions_labels):
    tags_list[i] = ' '.join(map(str, tags))

final_data = [[filename.split(".")[0], tags] for filename, tags in zip(file_names, tags_list)]
final_df = pd.DataFrame(final_data, columns=['image_name', 'tags'])
final_df.head()

timestr = time.strftime("%Y%m%d_%H%M%S")

final_df.to_csv('../submission_file_{}.csv'.format(timestr), index=False)