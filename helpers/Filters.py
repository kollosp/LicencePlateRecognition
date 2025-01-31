import math
import random

import cv2
import numpy as np
import random
from matplotlib import pyplot as plt
from sklearn.cluster import MeanShift, DBSCAN


hsv_limits = {
    "white": np.array([[0,0,100], [255,150,255]]),
    "black": np.array([[0,0,0], [255,255,50]]),
    "yellow": np.array([[20,0,0], [30,255,255]])
}

def gen_color(steps):
    m = np.zeros([1,1,3])
    m[0,0,0] = random.randint(100,255)
    m[0,0,1] = random.randint(100,255)
    m[0,0,2] = random.randint(100,255)
    return m.astype(np.uint8) #cv2.cvtColor(m, cv2.COLOR_HSV2BGR)

class Filters():
    def __init__(self):
        pass
    @staticmethod
    def histogram(image):
        histr = cv2.calcHist([image], [0], None, [256], [0, 256])
        plt.plot(histr)
        plt.xlim([0, 256])
        plt.show()

    @staticmethod
    def gaussian_blur(image, size=(9,9)):
        return cv2.GaussianBlur(image, size, 0)

    @staticmethod
    def sum_mask(masks):
        if len(masks) == 0: return np.zeros()
        for i in range(1,len(masks)):
            masks[0] = cv2.bitwise_or(masks[0], masks[i])
        return masks[0]
    @staticmethod
    def intersection_mask(masks):
        if len(masks) == 0: return np.zeros()
        for i in range(1,len(masks)):
            masks[0] = cv2.bitwise_and(masks[0], masks[i])
        return masks[0]

    @staticmethod
    def apply_mask(image, mask):
        return cv2.bitwise_and(image, image, mask=mask)

    @staticmethod
    def corner_detection_gray(image, k=0.04, blocksize = 2, ksize=3, kernel=5, blur_size=None):
        dst = cv2.cornerHarris(image, blocksize, ksize, k)
        #dst = cv2.dilate(dst, element)
        dst[dst > 0.001 * dst.max()] = 255
        dst[dst < 255] = 0
        element = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel, kernel))
        dst = cv2.dilate(dst, element).astype(np.uint8)

        if blur_size:
            dst = cv2.GaussianBlur(dst, blur_size, 0)
        # result is dilated for marking the corners, not important
        return dst


    @staticmethod
    def mean_shift_location(gray,eps=3):
        Z = np.zeros([gray.shape[0], gray.shape[1], 2])
        for i in range(gray.shape[0]):
            for j in range(gray.shape[1]):
                Z[i, j] = [i, j]

        Z = Z.reshape((-1, 2))

        #it requires more time than double loop?
        #additional = np.array([[i,j] for i in range(gray.shape[0]) for j in range(gray.shape[1])])
        indexes = gray.flatten() != 0
        Z = Z[indexes]
        labels = DBSCAN(eps=eps).fit_predict(Z)
        #c = np.zeros((labels.max()+1))
        #for i, _ in enumerate(c):
        #    c[i] = gen_color(i).mean()

        # print([np.count_nonzero(label == i) for i,_y in enumerate(center)], c)

        #coloring
        #res = np.zeros((gray.shape[0] * gray.shape[1]), np.uint8)
        #res[indexes] = labels * labels.max()-labels.min() / labels.max() - labels.min() * 255
        # res = res.astype(np.uint8).reshape(gray.shape)

        boxes = []
        for i in range(labels.max()):
            contour = Z[labels == i]
            y_max = contour[:,0].max()
            y_min = contour[:,0].min()
            x_max = contour[:,1].max()
            x_min = contour[:,1].min()
            boxes.append([x_min, y_min, x_max-x_min, y_max-y_min])


        return np.array(boxes).astype('int')


    @staticmethod
    def k_means_location(gray, K=8):
        Z = np.zeros([gray.shape[0],gray.shape[1],2])
        for i in range(gray.shape[0]):
            for j in range(gray.shape[1]):
                Z[i,j] = [i,j]

        Z = Z.reshape((-1,2))
        indexes = (gray.flatten() != 0)
        # convert to np.float32
        Z = np.float32(Z[indexes])

        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        ret, label, center = cv2.kmeans(Z, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
        center = np.uint8(center)
        c = np.zeros((K))
        label = label.flatten()
        for i, _ in enumerate(c):
            c[i] = gen_color(i).mean()

        # print([np.count_nonzero(label == i) for i,_y in enumerate(center)], c)

        res = np.zeros((gray.shape[0] * gray.shape[1]), np.uint8)
        res[indexes] = c[label]
        res = res.astype(np.uint8).reshape(gray.shape)
        return res

    @staticmethod
    def k_means(image, K = 8):
        additional = np.zeros([image.shape[0],image.shape[1],2])
        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                additional[i,j] = [i,j]

        Z = np.append(image, additional, axis=2).reshape((-1, 5))
        #Z = image.reshape((-1, 3))
        indexes = (Z != 0)[:,0]
        Z = Z[indexes]

        # convert to np.float32
        Z = np.float32(Z)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        ret, label, center = cv2.kmeans(Z, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
        center = np.uint8(center)
        c = np.zeros((K,3))
        label = label.flatten()

        for i, _ in enumerate(c):
            c[i] = gen_color(i)

        #print([np.count_nonzero(label == i) for i,_y in enumerate(center)], c)

        res = np.zeros((image.shape[0]*image.shape[1], 3), np.uint8)
        res[indexes] = c[label]
        res = res.astype(np.uint8).reshape(image.shape)
        return res

    @staticmethod
    def edge_filter(image,threshold1=100, threshold2=200, dilate_size=5):
        out = cv2.GaussianBlur(image, (5,5), 0)
        element = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate_size, dilate_size))
        out = cv2.Canny(image=out, threshold1=threshold1, threshold2=threshold2)
        return cv2.dilate(out, element)

    @staticmethod
    def variance_filter(image, element=cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9)), adaptive_range=15):
        mask = Filters.variance_filter_gray(image[:,:,0], element, adaptive_range)
        mask = cv2.bitwise_or(mask, Filters.variance_filter_gray(image[:,:,1], element, adaptive_range))
        mask = cv2.bitwise_or(mask, Filters.variance_filter_gray(image[:,:,2], element, adaptive_range))
        return mask

    @staticmethod
    def variance_filter_gray(image, element = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9,9)), adaptive_range=15):

        out = image.copy()
        kernel = np.array([[0, -1, 0],
                           [-1, 5, -1],
                           [0, -1, 0]])
        #out = cv2.filter2D(src=out, ddepth=-1, kernel=kernel)
        out = cv2.morphologyEx(out, cv2.MORPH_BLACKHAT, element) + cv2.morphologyEx(out, cv2.MORPH_TOPHAT, element)
        maxima = cv2.dilate(out, element)
        out = cv2.GaussianBlur(maxima, element.shape, 0)
        out = cv2.adaptiveThreshold(out, 255,
                                       cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, adaptive_range, 0)
        element = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        out = cv2.erode(out, element)
        out = cv2.dilate(out, element)
        return out

    @staticmethod
    def hsv_range_filter(bgr_image, hsv_l):
        hsv_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv_image, hsv_l[0], hsv_l[1])
        return cv2.bitwise_and(bgr_image, bgr_image, mask=mask)

    @staticmethod
    def hsv(name):
        return hsv_limits[name]

    @staticmethod
    def object_detection(image_gray, en_bounding_box=True, approximation_d=20, min_area = 0):
        #edges = cv2.Canny(image_gray, 5, 200)
        _, edges = cv2.threshold(image_gray,120,255,cv2.THRESH_BINARY)
        return Filters.object_detection_bin(edges, en_bounding_box=en_bounding_box, approximation_d=approximation_d, min_area=min_area)

    @staticmethod
    def object_detection_bin(image_bin, en_bounding_box=True, approximation_d=20, min_area = 0):
        # self.image_processing_steps_.append(out)
        # for 3.4 opencv version
        # _, contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # for 4.5 opencv version
        contours, _ = cv2.findContours(image_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        approx = []
        for i, contour in enumerate(contours):
            if en_bounding_box:
                x,y ,w,h = cv2.boundingRect(contour)
                if w*h > min_area:
                    approx.append([x,y,w,h])
            else:
                c = cv2.approxPolyDP(contour, approximation_d, True)
                approx.append(c)

        return approx


    @staticmethod
    def extract_rois(image, rois):
        images = []
        for r in rois:
            print(r[0],r[0]+r[2], r[1],r[1]+r[3])
            images.append(image[r[1]:r[1]+r[3], r[0]:r[0]+r[2]])
        return images

    @staticmethod
    def skeleton(image_bin, element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))):
        th = image_bin.copy()

        size = np.size(th)
        skel = np.zeros(th.shape, np.uint8)

        j = 0
        while True:
            j += 1
            # Step 2: Open the image
            open = cv2.morphologyEx(th, cv2.MORPH_OPEN, element)
            # Step 3: Substract open from the original image
            temp = cv2.subtract(th, open)
            # Step 4: Erode the original image and refine the skeleton
            eroded = cv2.erode(th, element)
            skel = cv2.bitwise_or(skel, temp)
            th = eroded.copy()
            # Step 5: If there are no white pixels left ie.. the image has been completely eroded, quit the loop
            if cv2.countNonZero(th) == 0 or j == 1000:
                break

        return skel