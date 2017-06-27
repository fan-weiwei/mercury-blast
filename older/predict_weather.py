"""
*** FOR EXAMPLE PURPOSES ***
"""
'''
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

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
    img = cv2.imread('../test-jpg-medium/{}.jpg'.format(name))
    x_test.append(img)

x_data = np.array(x_test, np.float16) / 255.

print(x_data.shape)

print("*** Loading Model ***")
# load json and create model
json_file = open('model-weather.json', 'r')
model_json = json_file.read()
json_file.close()
model = model_from_json(model_json)

model.load_weights("model-weather.h5")

print("*** Compiling Model ***")
model.compile(loss='categorical_crossentropy',
              # We NEED binary here, since categorical_crossentropy l1 norms the output before calculating loss.
              optimizer='adam',
              metrics=['accuracy'])

print("*** Generating Predictions ***")
#scores = model.evaluate(x_train, y_train, verbose=0)
#print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

predictions = model.predict(x_data)


print(*np.around(predictions[:10], decimals=3), sep='\n')

predictions_label = []
for prediction in predictions:
    #max_value = max(prediction)
    #max_index = prediction.index(max_value)
    max_index = np.argmax(prediction)
    label = models.weather_inv_map[max_index]
    predictions_label.append(label)

print(*predictions_label[:10], sep='\n')

final_data = [[filename.split(".")[0], tags] for filename, tags in zip(file_names, predictions_label)]
final_df = pd.DataFrame(final_data, columns=['image_name', 'tags'])
final_df.head()

final_df.to_csv('../partial/weather_submission_file.csv', index=False)

'''