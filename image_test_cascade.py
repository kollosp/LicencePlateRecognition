from __future__ import print_function
import cv2 as cv
import numpy as np
import argparse
from os import listdir
from os.path import isfile, join

def vconcat_resize_min(im_list, interpolation=cv.INTER_CUBIC):
    w_min = min(im.shape[1] for im in im_list)
    im_list_resize = [cv.resize(im, (w_min, int(im.shape[0] * w_min / im.shape[1])), interpolation=interpolation)
                      for im in im_list]
    return cv.vconcat(im_list_resize)

def hconcat_resize_min(im_list, interpolation=cv.INTER_CUBIC):
    h_min = min(im.shape[0] for im in im_list)
    im_list_resize = [cv.resize(im, (int(im.shape[1] * h_min / im.shape[0]), h_min), interpolation=interpolation)
                      for im in im_list]
    return cv.hconcat(im_list_resize)

def concat_tile_resize(im_list_2d, interpolation=cv.INTER_CUBIC):
    im_list_v = [hconcat_resize_min(im_list_h, interpolation=cv.INTER_CUBIC) for im_list_h in im_list_2d]
    return vconcat_resize_min(im_list_v, interpolation=cv.INTER_CUBIC)

## [main]
def main():

    dir = 'cascade/positive2'
    images = [f for f in listdir(dir) if isfile(join(dir, f))]
    image_index = 0

    cascade = cv.CascadeClassifier()
    cascade.load(cv.samples.findFile("cascade/out/cascade.xml"))

    while True:
        src = cv.imread(dir + "/" + images[image_index])
        #src = cv.rotate(src, cv.cv2.ROTATE_90_COUNTERCLOCKWISE)
        rects = cascade.detectMultiScale(src)

        for _, rect in enumerate(rects):
            print(rect)
            x, y, w, h = rect
            cv.rectangle(src,(x,y),(x+w,y+h),(0,255,0),2)

        cv.imshow("Haar Cascade",cv.resize(src, (int(src.shape[1]/2), int(src.shape[0]/2))))

        key = cv.waitKey(0)
        if key == ord('q') or key == 27:
            break
        elif key == ord('n'):
            image_index += 1
            print("image ", image_index, '/', len(images))
        elif key == ord('p'):
            image_index -= 1
            print("image ", image_index, '/', len(images))

## [main]
if __name__ == "__main__":

    main()
