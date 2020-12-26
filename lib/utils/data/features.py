import abc
import os

import cv2
import numpy as np


class DatasetFeature(metaclass=abc.ABCMeta):
    def __init__(self, folder_path, info):
        self.folder_path = folder_path
        self.info = info  # image_id, label

    @abc.abstractmethod
    def read(self, index):
        pass

    @property
    @abc.abstractmethod
    def name(self):
        pass


class RGBFrame(DatasetFeature):
    name = 'rgb_frame'

    def read(self, index):
        image_path = os.path.join(self.folder_path, self.info.iloc[index]['image_id'])
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = np.moveaxis(image, -1, 0) / 255.
        return image


class DiseaseLabel(DatasetFeature):
    name = 'disease_label'

    def read(self, index):
        label = self.info.iloc[index]['label']
        return label
