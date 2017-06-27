"""
*** FOR EXAMPLE PURPOSES ***
"""
'''
import os

import numpy as np  # linear algebra
from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense
from keras.models import Sequential, model_from_json

import models
import preprocessor

os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

from tqdm import tqdm

x_train = []
x_test = []
y_train = []

data = preprocessor.read_data('../train.csv')

print('*** Importing images ***')

for item in tqdm(data):
    x_train.append(models.AnnotatedRecord.medium_image(item))
    y_train.append(models.AnnotatedRecord.one_hot_weather(item))

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
'''
model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=(64, 64, 3)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(4, activation='softmax'))
'''

'''


print("*** Loading Model ***")
# load json and create model
json_file = open('model-weather.json', 'r')
model_json = json_file.read()
json_file.close()
model = model_from_json(model_json)

model.load_weights("model-weather.h5")


print("*** Saving Old Model ***")
# Serialize to JSON
model_json = model.to_json()
with open("model-weather-old.json", "w") as json_file:
    json_file.write(model_json)

model.save_weights("model-weather-old.h5")

print("*** Compiling Model ***")
model.compile(loss='categorical_crossentropy',
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
with open("model-weather.json", "w") as json_file:
    json_file.write(model_json)

model.save_weights("model-weather.h5")
print("Saved model to disk")



#from sklearn.metrics import fbeta_score

#p_valid = model.predict(x_valid, batch_size=128)
#print(y_valid)
#print(p_valid)
#print(fbeta_score(y_valid, np.array(p_valid) > 0.2, beta=2, average='samples'))


'''
