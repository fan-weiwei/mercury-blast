from tqdm import tqdm
from typing import List

from spectral import *
from skimage import io
from sklearn.preprocessing import MinMaxScaler
from PIL import Image
from clint.textui import *

from record import *
from config import *
import os
import numpy as np
import shutil
import pandas as pd

def create_spectral(source_path, target_path, size):

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
        im.thumbnail((size, size))
        im.save(target_path + '/' + tif.replace('tif', 'jpg'), 'JPEG', quality=95)

def create_visible(source_path, target_path, size):
    if not os.path.exists(target_path):
        os.makedirs(target_path)
    else:
        puts(colored.cyan('exists - {}'.format(target_path)))
        return False

    jpgs = os.listdir(source_path)

    for jpg in tqdm(jpgs):
        if 'jpg' not in jpg: continue
        im = Image.open(source_path + '/' + jpg)
        if size != 256:
            im.thumbnail((size, size))
        im.save(target_path + '/' + jpg, "JPEG")

def create_current_split(source_path, target_path, size):

    if not os.path.exists(target_path):
        os.makedirs(target_path)
    else:
        puts(colored.cyan('exists - {}'.format(target_path)))
        return False

    train_path = target_path + '/train'
    validate_path = target_path + '/validate'

    if not os.path.exists(train_path):
        os.makedirs(train_path)
    else:
        puts(colored.cyan('exists - {}'.format(train_path)))
        return False

    if not os.path.exists(validate_path):
        os.makedirs(validate_path)
    else:
        puts(colored.cyan('exists - {}'.format(validate_path)))
        return False

    jpgs = os.listdir(source_path)

    data = read_data(train_csv_path)
    training_data = data[:30000]
    validation_data = data[30000:]

    train_image_names = [x.name + '.jpg' for x in training_data]
    validation_image_names = [x.name + '.jpg' for x in validation_data]

    for jpg in tqdm(jpgs):
        if 'jpg' not in jpg: continue

        im = Image.open(source_path + '/' + jpg)
        if size != 256:
            im.thumbnail((size, size))

        if jpg in train_image_names:
            im.save(train_path + '/' + jpg, "JPEG")

        if jpg in validation_image_names:
            im.save(validate_path + '/' + jpg, "JPEG")


def read_data(path: str) -> List[object]:
    in_data = []
    with open(path, 'r') as f:
        raw = f.read()
        lines = raw.splitlines()
        records = list(map(lambda x: x.split(','), lines[1:]))
        for rec in records:
            name = rec[0]
            annotations = rec[1].split(' ')
            in_data.append(AnnotatedRecord(name, annotations))
    return in_data

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




