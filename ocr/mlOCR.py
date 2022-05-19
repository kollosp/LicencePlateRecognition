
import cv2
import numpy as np
import skimage.filters as filters
from skimage.morphology import rectangle
from scipy import ndimage
import matplotlib.pyplot as plt
from Vision import Vision
from helpers.Processing import Processing
from helpers.Filters import Filters

class MlOCR():
    def __init__(self, model, grid_size=(11,11), verbose=False):
        self.model = model
        self.verbose = verbose
        self.grid_size = grid_size

    def image_to_features(self, image):
        return (cv2.resize(image, self.grid_size)).flatten()

    def fit(self, signs, y):
        X = [self.image_to_features(sign) for sign in signs]
        self.model.fit(X, y)

        if self.model != "Dummy":
            self.model.fit(X, y)

        return self.model

    def predict(self, signs):
        X = [self.image_to_features(sign) for sign in signs]

        if self.model == "Dummy":
            return ['A' for x in X]
        else:
            return self.model.predict(X)

