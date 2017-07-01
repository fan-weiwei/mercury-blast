from tqdm import tqdm
from typing import List

from spectral import *
from skimage import io
from sklearn.preprocessing import MinMaxScaler
from PIL import Image
from clint.textui import *

import models
from config import *
import os
import numpy as np
import shutil
import pandas as pd

def create_spectral64(source_path, target_path):

    if not os.path.exists(target_path):
        os.makedirs(target_path)
    else:
        puts(colored.cyan('exists - {}'.format(target_path)))
        return False

    tifs = os.listdir(source_path)

    for tif in tqdm(tifs):
        if 'tif' not in tif: continue
        img = io.imread(source_path + '/' + tif)
        img3 = get_rgb(img, [3, 2, 1])  # NIR-R-G

        rescale_img = np.reshape(img3, (-1, 1))
        scaler = MinMaxScaler(feature_range=(0, 255))
        rescale_img = scaler.fit_transform(rescale_img)  # .astype(np.float32)
        img3_scaled = (np.reshape(rescale_img, img3.shape)).astype(np.uint8)

        im = Image.fromarray(img3_scaled)
        im.thumbnail((64, 64))
        im.save(target_path + '/' + tif.replace('tif', 'jpg'), 'JPEG', quality=95)




def read_data(path: str) -> List[object]:
    in_data = []
    with open(path, 'r') as f:
        raw = f.read()
        lines = raw.splitlines()
        records = list(map(lambda x: x.split(','), lines[1:]))
        for rec in records:
            name = rec[0]
            annotations = rec[1].split(' ')
            in_data.append(models.AnnotatedRecord(name, annotations))
    return in_data

def convert_jpg_to_medium():
    data = read_data(train_csv_path)

    for record in tqdm(data):
        im = Image.open('{}/{}.jpg'.format(original_train_image_path, record.name))
        im.thumbnail((64, 64))
        im.save('{}/{}.jpg'.format(medium_train_image_path, record.name), "JPEG")

    print(*data[:10], sep='\n')
    print(models.mapping)

def test_file_names():
    data = []
    with open(sample_submission_path, 'r') as f:
        raw = f.read()
        lines = raw.splitlines()
        records = list(map(lambda x: x.split(','), lines[1:]))
        for rec in records:
            name = rec[0]
            data.append(name)
    return data

def train_file_names():
    data = []
    with open(sample_submission_path, 'r') as f:
        raw = f.read()
        lines = raw.splitlines()
        records = list(map(lambda x: x.split(','), lines[1:]))
        for rec in records:
            name = rec[0]
            data.append(name)
    return data

def convert_test_jpg_to_small():
    data = []
    with open(sample_submission_path, 'r') as f:
        raw = f.read()
        lines = raw.splitlines()
        records = list(map(lambda x: x.split(','), lines[1:]))
        for rec in records:
            name = rec[0]
            data.append(name)

    for record in tqdm(data):
        try:
            im = Image.open('{}/{}.jpg'.format(original_test_image_path, record.name))
            im.thumbnail((32, 32))
            im.save('{}/{}.jpg'.format(small_test_image_path, record.name), "JPEG")
        except IOError:
            pass

        try:
            im = Image.open('{}/{}.jpg'.format(original_test_image_path, record.name))
            im.thumbnail((32, 32))
            im.save('{}/{}.jpg'.format(small_test_image_path, record.name), "JPEG")
        except IOError:
            pass

    print(*data[:10], sep='\n')
    print(models.mapping)

def convert_test_jpg_to_medium():
    data = []
    with open(config.sample_submission_path, 'r') as f:
        raw = f.read()
        lines = raw.splitlines()
        records = list(map(lambda x: x.split(','), lines[1:]))
        for rec in records:
            name = rec[0]
            data.append(name)

    for record in tqdm(data):
        try:
            im = Image.open('{}/{}.jpg'.format(original_test_image_path, record.name))
            im.thumbnail((64, 64))
            im.save('{}/{}.jpg'.format(medium_test_image_path, record.name), "JPEG")
        except IOError:
            pass

        try:
            im = Image.open('{}/{}.jpg'.format(original_test_image_path, record.name))
            im.thumbnail((64, 64))
            im.save('{}/{}.jpg'.format(medium_test_image_path, record.name), "JPEG")
        except IOError:
            pass

    print(*data[:10], sep='\n')
    print(models.mapping)


# im.show()
# print(im.format, im.size, im.mode)




