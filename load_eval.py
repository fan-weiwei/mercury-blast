import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

from keras.models import model_from_json
import numpy as np  # linear algebra
from tqdm import tqdm
import preprocessor
import models

x_train = []
x_test = []
y_train = []

data = preprocessor.read_data('../train.csv')

print('*** Importing Images ***')

for item in tqdm(data):
    x_train.append(models.AnnotatedRecord.small_image(item))
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

print("*** Evaluating Model ***")
scores = model.evaluate(x_valid, y_valid, verbose=0)
print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
