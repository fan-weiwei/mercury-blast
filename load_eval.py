from keras.models import model_from_json
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import cv2
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


# load json and create model
json_file = open('model.json', 'r')
model_json = json_file.read()
json_file.close()
model = model_from_json(model_json)

model.load_weights("model.h5")
print("Loaded model from disk")

model.compile(loss='binary_crossentropy',
              # We NEED binary here, since categorical_crossentropy l1 norms the output before calculating loss.
              optimizer='adam',
              metrics=['accuracy'])


scores = model.evaluate(x_train, y_train, verbose=0)
print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
