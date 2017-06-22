import cv2
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Dense, Dropout, Flatten
from keras.models import Sequential
from tqdm import tqdm

x_train = []
x_test = []
y_train = []

df_train = pd.read_csv('../train.csv')

flatten = lambda l: [item for sublist in l for item in sublist]
labels = list(set(flatten([l.split(' ') for l in df_train['tags'].values])))

label_map = {l: i for i, l in enumerate(labels)}
inv_label_map = {i: l for l, i in label_map.items()}

for f, tags in tqdm(df_train.values, miniters=1000):
    img = cv2.imread('../train-jpg/{}.jpg'.format(f))
    targets = np.zeros(17)
    for t in tags.split(' '):
        targets[label_map[t]] = 1
    x_train.append(cv2.resize(img, (32, 32)))
    y_train.append(targets)

y_data = np.array(y_train, np.uint8)
x_data = np.array(x_train, np.float16) / 255.

print(x_data.shape)
print(y_data.shape)

split = 35000
x_train = x_data[:split]
x_valid = x_data[split:]
y_train = y_data[:split]
y_valid = y_data[split:]


model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=(32, 32, 3)))

model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.50))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.8))
model.add(Dense(17, activation='sigmoid'))

model.compile(loss='binary_crossentropy',
              # We NEED binary here, since categorical_crossentropy l1 norms the output before calculating loss.
              optimizer='adam',
              metrics=['accuracy'])

model.fit(x_train, y_train,
          batch_size=128,
          epochs=1,
          verbose=1,
          validation_data=(x_valid, y_valid))
print("Done training")

scores = model.evaluate(x_train, y_train, verbose=0)
print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

# Serialize to JSON
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)

model.save_weights("model.h5")
print("Saved model to disk")



#from sklearn.metrics import fbeta_score
#
#p_valid = model.predict(x_valid, batch_size=128)
#print(y_valid)
#print(p_valid)
#print(fbeta_score(y_valid, np.array(p_valid) > 0.2, beta=2, average='samples'))



