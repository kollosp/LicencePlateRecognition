import os
#os.environ['DISPLAY'] = ':0'
import cv2
import numpy as np
import random
import time
from os import listdir
from os.path import isfile, join

start = time.time()
end = float('inf')

class Pipeline:
    def __init__(self, processors):
        self.processors = processors

    def __call__(self, *args, **kwargs):
        src = args[0]
        for i, processor in enumerate(self.processors):
            src = processor(src)
        return src


def contour_features_extraction(mat, contour):

    x = contour[:, 0, 0]
    y = contour[:, 0, 1]
    x_min = x.min()
    y_min = y.min()
    x_max = x.max()
    y_max = y.max()
    M = cv2.moments(contour)
    features = None
    area = cv2.contourArea(contour)
    area_box = (x_max - x_min) * (y_max - y_min)
    if M["m00"] > 0:
        features = {
            "x_min": x_min,
            "y_min": y_min,
            "width": x_max - x_min,
            "height": y_max - y_min,
            "ratio": (x_max - x_min) / (y_max - y_min),
            "mass_x": M["m10"] / M["m00"] - x_min,
            "mass_y": M["m01"] / M["m00"] - y_min,
            "area": area,
            "area_f": area / area_box,
            "area_box": area_box,
            "points": len(contour)
        }

    return features

def rotate(src):
    return cv2.rotate(src, cv2.ROTATE_90_COUNTERCLOCKWISE)


def process(src):
    w = src.shape[1]
    h = src.shape[0]

    dst = np.zeros([h, w*5, src.shape[2]], np.uint8)
    dst[:, 0:w] = src.copy()

    gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)

    dilation_size = 1
    dilated = cv2.dilate(src, cv2.getStructuringElement(cv2.MORPH_RECT, (2 * dilation_size + 1, 2 * dilation_size + 1), (dilation_size, dilation_size)))


    #_, thresh = cv2.threshold(gray,127,255,cv2.THRESH_BINARY)
    canny = cv2.Canny(dilated,50,120)

    #erode_size = 2
    #eroded = cv2.erode(canny, cv2.getStructuringElement(cv2.MORPH_RECT, (2 * erode_size + 1, 2 * erode_size + 1), (erode_size, erode_size)))

    #dilation_size_2 = 1
    #canny = cv2.dilate(canny, cv2.getStructuringElement(cv2.MORPH_RECT, (2 * dilation_size_2 + 1, 2 * dilation_size_2 + 1),
    #                                                    (dilation_size_2, dilation_size_2)))

    #dst[:, w:2*w] =  cv2.cvtColor(thresh, cv2.COLOR_GRAY2RGB)
    dst[:, w:2*w] =  dilated
    #dst[:, w:2*w] = cv2.cvtColor(eroded, cv2.COLOR_GRAY2RGB)
    dst[:, w*2:3*w] = cv2.cvtColor(canny, cv2.COLOR_GRAY2RGB)

    contours, hierarchy = cv2.findContours(canny, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    #filtering contours

    #contours_filtered = [x for x in contours if len(x) > 3 and len(x) < 10]

    con_image = src.copy()

    for index, contour in enumerate(contours):
        features = contour_features_extraction(src, contour)
        #remove contours without any area
        if features is not None:
            cv2.drawContours(con_image, [contour], -1, (0,0,random.randint(100,255)),5)
            x,y,bb_w,bb_h = cv2.boundingRect(contour)
            #cv2.putText(con_image, str(index), (x, y+15), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 4)
            #cv2.putText(con_image, str(index), (x, y+15), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            print(index, features)
            cv2.rectangle(con_image,(x,y),(x+bb_w,y+bb_h),(0,255,0),2)
            con_image = cv2.circle(con_image, (int(features["mass_x"] + features["x_min"]), int(features["mass_y"] + features["y_min"])),
                                   radius=3, color=(0, 0, 255), thickness=5)

    dst[:, w*3:4*w] = con_image

    return dst


def processOne():
    images = [
        '220219_data/20220219094916_35_camera_capture.png',
        '220219_data/20220219094926_36_camera_capture.png',
        '220219_data/20220219095027_42_camera_capture.png'
    ]

    pipeline = Pipeline([
        cv2.imread,
        rotate,
        process
    ])

    for i, image in enumerate(images):
        w_title = "test_{}".format(i)
        cv2.namedWindow(w_title)
        start = time.time()
        img = pipeline(image)
        cv2.imshow(w_title, cv2.resize(img, (int(img.shape[1] / 2), int(img.shape[0] / 2))))
        print("execution time: {0:.3f}".format(time.time() - start))

        cv2.waitKey(0)
    cv2.destroyAllWindows()
    exit()

def processStream():
    dir = '220218_data'
    images = [dir + "/" + f for f in listdir(dir) if isfile(join(dir, f))]
    print(images)
    pipeline = Pipeline([
        cv2.imread,
        rotate,
        process
    ])
    cv2.namedWindow("test")
    cv2.namedWindow("org")

    for i, image in enumerate(images):
        start = time.time()
        cv2.imshow("org", cv2.imread(image))
        img = pipeline(image)
        cv2.imshow("test", cv2.resize(img, (int(img.shape[1] / 2), int(img.shape[0] / 2))))
        print("execution time: {0:.3f}".format(time.time() - start))
        cv2.waitKey(0)

    cv2.destroyAllWindows()
    exit()

if __name__ == "__main__":
    processStream()
