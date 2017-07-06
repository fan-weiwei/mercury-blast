from abc import ABC, abstractmethod
import preprocessor
import cv2
import numpy as np

from tqdm import tqdm
from config import *
from record import *

class Model(ABC):
    @abstractmethod
    def training_images(self):
        pass


class Super128(Model):

    def __init__(self):
        self.training_data = preprocessor.read_data(train_csv_path)

    def training_images(self):

        print('*** Importing images ***')

        images = []

        for item in tqdm(self.training_data):
            image = cv2.imread(visible128_path + '/train/{}.jpg'.format(item.name))
            images.append(image)

        return np.array(images, np.float16) / 255.

    def training_some_hot(self):

        annotations = []

        for item in tqdm(self.training_data):
            targets = np.zeros(17)
            for t in item.annotations:
                targets[mapping[t]] = 1

            annotations.append(targets)

        return np.array(annotations, np.uint8)



current_model = Super128()

