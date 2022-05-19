
import cv2
import numpy as np
import skimage.filters as filters
from skimage.morphology import rectangle
from scipy import ndimage
import matplotlib.pyplot as plt
from Vision import Vision
from helpers.Processing import Processing
from helpers.Filters import Filters


class MlModel():
    def __init__(self, model, blur_size=(81,81), window_size=(9,9), verbose=False, image_verbose=True, use_sliding_window=False):
        self.model = model
        self.verbose = verbose
        self.image_verbose = image_verbose
        self.window_size = window_size
        self.blur_size = blur_size
        self.use_sliding_window = use_sliding_window
        self.first_fit = True

    def y_to_rects(self, Y, image_shape):
        shape = self.create_shape(image_shape)

        Y = Y.reshape(shape[0] , shape[1])
        YY = np.zeros((image_shape[0] , image_shape[1]), np.uint8)

        for y in range(shape[0]):
            for x in range(shape[1]):
                x_s, y_s = x,y
                if not self.use_sliding_window:
                    x_s = self.window_size[1] * x
                    y_s = self.window_size[0] * y
                YY[y_s:y_s+self.window_size[0], x_s:x_s+self.window_size[1]] = Y[y,x] * 255

        return YY


    def rects_to_y(self, rects, image_shape):
        shape = self.create_shape(image_shape)
        print(shape, image_shape)
        YY = np.zeros((image_shape[0], image_shape[1]), np.uint8)
        Y = np.zeros((shape[0], shape[1]), np.float)

        for rect in rects:
            YY[rect[1]:rect[1]+rect[3], rect[0]:rect[0]+rect[2]] = 255

        for y in range(shape[0]):
            for x in range(shape[1]):
                x_s, y_s = x,y
                if not self.use_sliding_window:
                    x_s = self.window_size[1] * x
                    y_s = self.window_size[0] * y
                #print(y_s,y_s+self.window_size[1], x_s,x_s+self.window_size[0],shape,image_shape)

                Y[y,x] = np.max((YY[y_s:y_s+self.window_size[1], x_s:x_s+self.window_size[0]]).flatten()) / 255.0

        #print("Y", Y.max())
        # if self.image_verbose:
        #     cv2.imshow("YY", YY)
        #     cv2.imshow("Y", Y)
        #     print("YY.flatten", YY.flatten().max(), rects)

        return Y.flatten()

    def image_to_features(self, image, is_learning=False):
        #features: gray,variance,corners
        window_size = self.window_size[0] * self.window_size[1]
        shape = self.create_shape(image.shape)

        element = cv2.getStructuringElement(cv2.MORPH_RECT, self.blur_size)

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        f = [
            #hsv[:,:,0],
            #hsv[:,:,2] -hsv[:,:,1],
            #Filters.variance_filter(image, element),
            cv2.GaussianBlur(Filters.edge_filter(gray, 80, 100, dilate_size=5), self.blur_size, 0),
            Filters.corner_detection_gray(gray, kernel=5, blur_size=self.blur_size),
            cv2.GaussianBlur(Filters.edge_filter(gray, 120, 160, dilate_size=5), self.blur_size, 0),
            Filters.corner_detection_gray(gray, blocksize=4, ksize=7, k=0.01, kernel=5, blur_size=self.blur_size)
        ]

        if self.image_verbose and is_learning:
            f_img = []
            for i, _ in enumerate(f):
                f_img.append(cv2.resize(f[i], (int(f[i].shape[1] / 2),  int(f[i].shape[1] / 2))))
            cv2.imshow("features", Vision.hconcat_resize_min(f_img))
            cv2.waitKey(2000)

        # features + global mean value
        features_additional = 1
        features_count = len(f) * window_size + features_additional
        features = np.zeros((shape[0], shape[1], features_count), np.float)

        image_mean = gray.flatten().mean()
        for i, _ in enumerate(f):
            ii = i*window_size
            for y in range(shape[0]):
                for x in range(shape[1]):
                    x_s, y_s = x,y
                    if not self.use_sliding_window:
                        x_s = self.window_size[1] * x
                        y_s = self.window_size[0] * y
                    #print(f[i][y_s:y_s+self.window_size[0], x_s:x_s+self.window_size[1]].flatten())
                    features[y,x,ii:ii+window_size] = (f[i][y_s:y_s+self.window_size[0], x_s:x_s+self.window_size[1]]).flatten() / 255.0
                    features[y,x,-1] = image_mean / 255.0

        # if self.image_verbose:
        #     for i in range(features_count):
        #         cv2.imshow("feature " + str(i), features[:,:,i])
        print("features_count", features_count)
        features = features.reshape((shape[0] * shape[1], features_count))
        return features

    def partial_fit(self, image, rectangles):
        if self.verbose:
            print("=== Fitting model ===")

        y = self.rects_to_y(rectangles, image.shape)
        X = self.image_to_features(image, is_learning=True)

        r = None
        if self.first_fit:
            r = self.fit_(X, y)
        else:
            r = self.partial_fit_(X, y)

        self.model.save("mlp_models")
        return r

    def fit_(self,X,y):
        self.first_fit = False
        self.model.fit(X, y)

    def partial_fit_(self,X,y):
        self.model.partial_fit(X, y)


    def predict(self, image):
        X = self.image_to_features(image)
        prediction = self.model.predict(X)



        return self.y_to_rects(prediction, image.shape)


    def predict_rois(self, image):
        X = self.image_to_features(image)
        prediction = self.model.predict(X)
        prediction = self.y_to_rects(prediction, image.shape)

        prediction = cv2.GaussianBlur(prediction, (81, 81), 0)

        f_pred = prediction.flatten()
        prediction = ((255 / f_pred.flatten().max()) * prediction).astype(np.uint8)

        rects = Filters.object_detection(prediction)
        return rects

    def create_shape(self, image_shape):

        if self.use_sliding_window:
            shape = (image_shape[0] - self.window_size[0], image_shape[1] - self.window_size[1])
        else:
            shape = (int(image_shape[0] / self.window_size[0]), int(image_shape[1] / self.window_size[1]))

        return shape