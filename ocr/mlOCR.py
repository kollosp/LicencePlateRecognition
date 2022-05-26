
import cv2
import numpy as np
import pickle
import skimage.filters as filters
from skimage.morphology import rectangle
from scipy import ndimage
import matplotlib.pyplot as plt
from Vision import Vision
from helpers.Processing import Processing
from helpers.Filters import Filters
from sklearn.metrics import balanced_accuracy_score

class MlOCR():
    def __init__(self, model= "Dummy", grid_size=(11,11), path=None, image_verbose = False, verbose=True, learning_iterations=100, metric=balanced_accuracy_score, alpha=0.25):

        self.model = model
        self.verbose = verbose
        self.learning_iterations = learning_iterations
        self.grid_size = grid_size
        self.metric = metric
        self.image_verbose = image_verbose
        self.alpha = alpha

        self.model_fitted = False

    def image_to_features(self, image):
        return (cv2.resize(image, self.grid_size)).flatten()

    def fit(self, signs, y):
        X = np.array([self.image_to_features(sign) for sign in signs])

        if self.model != "Dummy":
            for i in range(self.learning_iterations):
                if self.model_fitted:
                    self.model.partial_fit(X, y)
                else:
                    self.model.fit(X, y)
                    self.model_fitted = True

                if self.verbose:
                    p = self.model.predict(X)
                    score = self.metric(p, y)
                    print("Learning progress", i, "/", self.learning_iterations, "score:", score)

        return self.model

    def predict(self, signs):
        if len(signs) == 0:
            return []

        X = np.array([self.image_to_features(sign) for sign in signs])

        if self.model == "Dummy":
            return ['A' for x in X]
        else:
            predictions =  self.model.predict(X)
            pp =  self.model.predict_proba(X)


            for i, p, s in zip(range(len(predictions)), predictions, signs):
                if max(pp[i]) >= self.alpha:
                    if self.image_verbose:
                        cv2.imshow(str(i) + " " + p, cv2.resize(s, self.grid_size))
                    #print(str(i) + " " + p, "\n", pp[i])
                else:
                    predictions[i] = "-"
            return predictions


    def save(self, path):
        self.path = path
        # Its important to use binary mode
        dbfile = open(self.path , 'ab')

        # source, destination
        pickle.dump(self, dbfile)
        dbfile.close()


    @staticmethod
    def load(path):
        dbfile = open(path, 'rb')
        ss = pickle.load(dbfile)
        dbfile.close()
        return ss

    @staticmethod
    def shift_image( img, dx, dy):
        img = np.roll(img, dy, axis=0)
        img = np.roll(img, dx, axis=1)
        if dy > 0:
            img[:dy, :] = 255
        elif dy < 0:
            img[dy:, :] = 255
        if dx > 0:
            img[:, :dx] = 255
        elif dx < 0:
            img[:, dx:] = 255
        return img