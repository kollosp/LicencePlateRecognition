from __future__ import print_function
import cv2 as cv
import numpy as np
import argparse
from os import listdir
from os.path import isfile, join
import random

import Vision


color_max = np.array([180,50,140])
color_min = np.array([0,12,100])

click_param = {
    "x": 0,
    "y": 0,
    "click_count": 0,
    "boxes": [[]]
}

def onclick(event, x, y, flags, param):
    # left button
    if event == cv.EVENT_LBUTTONDOWN:

        # because images are scaled by 1/2 the real coordinate has to be multiplied by 2
        param["boxes"][-1].append([2*x,2*y])

    # middle button
    if event == cv.EVENT_MBUTTONDOWN:
        param["boxes"].append([])

    # print(param["boxes"])


# [main]
def main():
    dir = 'saves'
    save_dir = 'cropped'
    images = [f for f in listdir(dir) if isfile(join(dir, f))]
    image_index = 0

    src = cv.imread(dir + "/" + images[image_index])
    if src is None:
        print('Could not open or find the image')
        exit(0)

    window_name = "Original"
    cv.namedWindow(window_name)
    cv.setMouseCallback(window_name, onclick, click_param)

    while True:

        contours = src.copy()

        for _, contour in enumerate(click_param["boxes"]):
            if len(contour) > 1:
                contour = np.array(contour)
                cv.drawContours(contours,[contour], -1, (0, 0, 255), 5)
                x, y, w, h = cv.boundingRect(contour)
                cv.rectangle(contours,(x,y),(x+w,y+h),(0,255,0),2)

        img = Vision.concat_tile_resize([[src, contours]])

        # scaling by 1/2
        cv.imshow(window_name,cv.resize(img, (int(img.shape[1]/2), int(img.shape[0]/2))))

        key = cv.waitKey(30)
        if key == ord('q') or key == 27:
            break
        elif key == ord('n'):
            image_index += 1
            src = cv.imread(dir + "/" + images[image_index])
            print("image ", image_index, '/', len(images))
        elif key == ord('p'):
            image_index -= 1
            src = cv.imread(dir + "/" + images[image_index])
            print("image ", image_index, '/', len(images))
        elif key == ord('s'):
            name, ext = images[image_index].split(".")

            for i, contour in enumerate(click_param["boxes"]):
                if len(contour) > 1:
                    contour = np.array(contour)
                    x, y, w, h = cv.boundingRect(contour)
                    cv.rectangle(contours, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    file_name = save_dir + '/' + name + "_" + str(i) + "." + ext
                    cv.imwrite(file_name, src[y:y+h,x:x+w])
                    print("image saved in", file_name)
            click_param["boxes"] = [[]]


## [main]
if __name__ == "__main__":

    main()