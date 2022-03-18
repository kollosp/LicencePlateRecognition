import cv2
import numpy as np
import skimage.filters as filters
from skimage.morphology import rectangle
from scipy import ndimage
import matplotlib.pyplot as plt

def std(array):
    return array.flatten().std()


def max_min(array):
    return array.flatten().max() - array.flatten().min()

def max(array):
    return array.flatten().max()

def min(array):
    return array.flatten().max()


class ContourFinder:
    def __init__(self, file=None, canny_1=50, canny_2=200, dilation_kernel_size=1, approximation_d=20):
        self.file = file
        self.canny_1 = canny_1
        self.canny_2 = canny_2
        self.approximation_d = approximation_d
        self.dilation_kernel_size = dilation_kernel_size

        self.image_processing_steps_ = []
        self.feature_set_ = []
        self.label_set_ = []

        if self.file is not None:
            self.load_from_file(self.file)
        pass

    def process_2d(self, image, kernel, funcs):
        res = np.zeros((len(funcs), image.shape[0], image.shape[1]))
        kernel_y = int(len(kernel) / 2)
        kernel_x = int(len(kernel[0]) / 2)
        image_y = len(image)
        image_x = len(image[0])
        for i in range(kernel_x, image_x - kernel_x):
            for j in range(kernel_y, image_y - kernel_y):
                for k, func in enumerate(funcs):
                    #s = image[j-kernel_y:j+kernel_y, i-kernel_x:i+kernel_x, :]
                    s = image[j-kernel_y:j+kernel_y, i-kernel_x:i+kernel_x]
                    res[k, j, i] = func(s)
        return res

    def box_to_contour(self, box):
        return np.array([
            [box[0], box[1]],
            [box[0]+box[2], box[1]],
            [box[0]+box[2], box[1]+box[3]],
            [box[0], box[1]+box[3]],
        ])

    def contour_intersect(self, image, contour1, contour2):
        # Two separate contours trying to check intersection on
        contours = [contour1, contour2]

        # Create image filled with zeros the same size of original image
        blank = np.zeros(image.shape[0:2])

        # Copy each contour into its own image and fill it with '1'
        image1 = cv2.drawContours(blank.copy(), contours, 0, 1)
        image2 = cv2.drawContours(blank.copy(), contours, 1, 1)

        # Use the logical AND operation on the two images
        # Since the two images had bitwise and applied to it,
        # there should be a '1' or 'True' where there was intersection
        # and a '0' or 'False' where it didnt intersect
        intersection = np.logical_and(image1, image2)

        # Check if there was a '1' in the intersection
        return intersection.any()

    def contour_features_extraction(self, image, contour):
        x = contour[:, 0, 0]
        y = contour[:, 0, 1]
        x_min = x.min()
        y_min = y.min()
        x_max = x.max()
        y_max = y.max()
        box = cv2.boundingRect(contour)
        M = cv2.moments(contour)
        features = None
        area = cv2.contourArea(contour)
        area_box = (x_max - x_min) * (y_max - y_min)
        #if M["m00"] > 0:
        features = {
            "x_min": x_min,
            "y_min": y_min,
            "width": x_max - x_min,
            "height": y_max - y_min,
            "ratio": (x_max - x_min) / (y_max - y_min),
            #"mass_x": M["m10"] / M["m00"] - x_min,
            #"mass_y": M["m01"] / M["m00"] - y_min,
            "area": area,
            #"area_f": area / (box[2]*box[3]),
            "area_box": area_box,
            "points": len(contour),
        }

        return features

    def fit(self, image, true_bounding_boxes):

        contours = self.find_contours(image)
        rects = []

        for i, contour in enumerate(contours):
            features = self.contour_features_extraction(image, contour)
            if features is not None:
                rects.append(contour)
                self.feature_set_.append(features)
                for j, true_values in enumerate(true_bounding_boxes):
                    if self.contour_intersect(image, contour, self.box_to_contour(true_values)):
                        # given set of features is a positive one
                        print("Contour intersect plate.", features)
                        self.label_set_.append(1)

                    else:
                        # given set of features is an negative one
                        self.label_set_.append(0)
                        pass

        return rects

    def export_features(self, file):
        array = np.zeros([len(self.feature_set_),11+1])

        for i, _ in enumerate(self.feature_set_):
            print(self.feature_set_[i])
            array[i, 0] = self.feature_set_[i]['x_min']
            array[i, 1] = self.feature_set_[i]['y_min']
            array[i, 2] = self.feature_set_[i]['width']
            array[i, 3] = self.feature_set_[i]['height']
            array[i, 4] = self.feature_set_[i]['ratio']
            #array[i, 5] = self.feature_set_[i]['mass_x']
            #array[i, 6] = self.feature_set_[i]['mass_y']
            array[i, 7] = self.feature_set_[i]['area']
            #array[i, 8] = self.feature_set_[i]['area_f']
            array[i, 9] = self.feature_set_[i]['area_box']
            array[i, 10] = self.feature_set_[i]['points']
            array[i, 11] = self.label_set_[i]

        print(array)


    def load_from_file(self, file):
        pass

    def find_contours(self, image):
        self.image_processing_steps_ = []
        kernel5x5 = np.ones([8,8])
        print(kernel5x5)
        out = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        #out2 = self.process_2d(out, kernel5x5, [std, max_min, max, min])
        #for i, img in enumerate(out2):
        #    self.image_processing_steps_.append(img.astype(np.uint8))

        out2 = ndimage.filters.generic_filter(out, np.std, (8,8))
        self.image_processing_steps_.append(out2)
        #_, out2 = cv2.threshold(out, out2.flatten().mean(), 255, cv2.THRESH_BINARY)

        flatten = out2.flatten()
        th_level = (flatten.max()-flatten.min())/3
        print("image mean:", len(flatten[flatten > 0]))
        print("image treshold:", th_level)
        print("image max:", flatten.max())
        print("image min:", flatten.min())

        #fig, axs = plt.subplots(1, 2, sharey=True, tight_layout=True)
        #plt.hist(out2.flatten(), bins=255)
        #plt.show()

        _, out2 = cv2.threshold(out2, th_level, 255, cv2.THRESH_BINARY)
        self.image_processing_steps_.append(out2)

        out = cv2.Canny(out, self.canny_1, self.canny_2)
        self.image_processing_steps_.append(out)

        if self.dilation_kernel_size is not None:
            dilation_size = self.dilation_kernel_size
            out = cv2.dilate(out,cv2.getStructuringElement(cv2.MORPH_RECT, (2 * dilation_size + 1, 2 * dilation_size + 1),
                                                           (dilation_size, dilation_size)))
            out = cv2.erode(out,
                            cv2.getStructuringElement(cv2.MORPH_RECT, (2 * dilation_size + 1, 2 * dilation_size + 1),
                                                      (dilation_size, dilation_size)))

            self.image_processing_steps_.append(out)
            out = cv2.bitwise_and(out, out2)
            self.image_processing_steps_.append(out)

        #out = cv2.morphologyEx(out, cv2.MORPH_OPEN, kernel)
        #out = cv2.filter2D(out, -1, kernel5x5)


        #self.image_processing_steps_.append(out)
        # for 3.4 opencv version
        _, contours, _ = cv2.findContours(out, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # for 4.5 opencv version
        #contours, _ = cv2.findContours(out, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        print(len(contours))
        approx = []
        for i, contour in enumerate(contours):
            #epsilon = 0.1 * cv2.arcLength(contour, True)
            approx.append(cv2.approxPolyDP(contour, self.approximation_d, True))

        return approx

    def detectMultiScale(self, image):

        contours = self.find_contours(image)
        rects = []
        for index, contour in enumerate(contours):
            features = self.contour_features_extraction(image, contour)

            #predict features !!!

            rects.append(cv2.boundingRect(contour))

        return rects