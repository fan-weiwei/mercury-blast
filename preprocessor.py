#from PIL import Image
#im = Image.open('../train-jpg/train_101.jpg')
#im.show()

#import numpy as np  # linear algebra
#import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
#from tqdm import tqdm

import models
x_train = []
x_test = []
y_train = []

data = []

with open('../train.csv', 'r') as f:
    read_data = f.read()
    lines = read_data.splitlines()
    records = list(map(lambda x: x.split(','), lines[1:]))
    for record in records:
        name = record[0]
        annotations = record[1].split(' ')
        data.append(models.AnnotatedRecord(name, annotations))



    print(data[0])


'''
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
'''