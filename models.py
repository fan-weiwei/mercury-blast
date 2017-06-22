from typing import List
import numpy as np
import cv2

class AnnotatedRecord:

    def __init__(self, name: str, annotations: List[str]) -> object:
        self.name = name
        self.annotations = annotations

    def __repr__(self):
        return "<AnnotatedRecord name:%s annotations:%s>" % (self.name, self.annotations)

    def __str__(self):
        return "%s, %s" % (self.name, self.annotations)

    def some_hot(self) -> object:
        targets = np.zeros(17)
        for t in self.annotations:
            targets[mapping[t]] = 1
        return targets

    def small_image(self):
        return cv2.imread('../train-jpg-small/{}.jpg'.format(self.name))

mapping = {

    # land labels, 1+
    'primary' : 0,
    'agriculture' : 1,
    'water' : 2,
    'cultivation' : 3,
    'habitation' : 4,
    'road': 5,

    # weather labels, exactly 1
    'clear' : 6,
    'partly_cloudy' : 7,
    'haze' : 8,
    'cloudy' : 9,

    # rare, usually 1
    'slash_burn' : 10,
    'conventional_mine' : 11,
    'bare_ground' : 12,
    'artisinal_mine' : 13,
    'blooming' : 14,
    'selective_logging' : 15,
    'blow_down' : 16,

}

inv_mapping = {i: l for l, i in mapping.items()}