from PIL import Image
from tqdm import tqdm
from typing import List

import models
import config

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
    data = read_data(config.train_csv_path)

    for record in tqdm(data):
        im = Image.open('{}/{}.jpg'.format(config.original_train_image_path, record.name))
        im.thumbnail((64, 64))
        im.save('{}/{}.jpg'.format(config.medium_train_image_path, record.name), "JPEG")

    print(*data[:10], sep='\n')
    print(models.mapping)

def test_file_names():
    data = []
    with open(config.sample_submission_path, 'r') as f:
        raw = f.read()
        lines = raw.splitlines()
        records = list(map(lambda x: x.split(','), lines[1:]))
        for rec in records:
            name = rec[0]
            data.append(name)
    return data

def train_file_names():
    data = []
    with open(config.sample_submission_path, 'r') as f:
        raw = f.read()
        lines = raw.splitlines()
        records = list(map(lambda x: x.split(','), lines[1:]))
        for rec in records:
            name = rec[0]
            data.append(name)
    return data

def convert_test_jpg_to_small():
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
            im = Image.open('{}/{}.jpg'.format(config.original_test_image_path_one, record.name))
            im.thumbnail((32, 32))
            im.save('{}/{}.jpg'.format(config.small_test_image_path, record.name), "JPEG")
        except IOError:
            pass

        try:
            im = Image.open('{}/{}.jpg'.format(config.original_test_image_path_two, record.name))
            im.thumbnail((32, 32))
            im.save('{}/{}.jpg'.format(config.small_test_image_path, record.name), "JPEG")
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
            im = Image.open('{}/{}.jpg'.format(config.original_test_image_path_one, record.name))
            im.thumbnail((64, 64))
            im.save('{}/{}.jpg'.format(config.medium_test_image_path, record.name), "JPEG")
        except IOError:
            pass

        try:
            im = Image.open('{}/{}.jpg'.format(config.original_test_image_path_two, record.name))
            im.thumbnail((64, 64))
            im.save('{}/{}.jpg'.format(config.medium_test_image_path, record.name), "JPEG")
        except IOError:
            pass

    print(*data[:10], sep='\n')
    print(models.mapping)


# im.show()
# print(im.format, im.size, im.mode)




