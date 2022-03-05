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


color_max = np.array([180,50,140])
color_min = np.array([0,12,100])

def hue_max(val):
    color_max[0] = val
def hue_min(val):
    color_min[0] = val
def sat_max(val):
    color_max[1] = val
def sat_min(val):
    color_min[1] = val
def val_max(val):
    color_max[2] = val
def val_min(val):
    color_min[2] = val
## [main]
def main():

    print("Use:")
    print(" - 's' to save and image")
    print(" - 'n' to display next image")
    print(" - 'p' to display previous  image")

    dir = '220219_data'
    save_dir = 'saves'
    images = [f for f in listdir(dir) if isfile(join(dir, f))]
    image_index = 0

    src = cv.imread(dir + "/" + images[image_index])
    src = cv.rotate(src, cv.cv2.ROTATE_90_COUNTERCLOCKWISE)
    if src is None:
        print('Could not open or find the image')
        exit(0)

    window_name = "HSV range"
    cv.namedWindow(window_name)
    cv.createTrackbar("Hue max",window_name, 0, 180, hue_max)
    cv.createTrackbar("Hue min",window_name, 0, 255, hue_min)
    cv.createTrackbar("Saturation max",window_name, 0, 255, sat_max)
    cv.createTrackbar("Saturation min",window_name, 0, 255, sat_min)
    cv.createTrackbar("Value max",window_name, 0, 255, val_max)
    cv.createTrackbar("Value min",window_name, 0, 255, val_min)

    cv.setTrackbarPos("Hue max", window_name, color_max[0])
    cv.setTrackbarPos("Hue min", window_name, color_min[0])
    cv.setTrackbarPos("Saturation max", window_name, color_max[1])
    cv.setTrackbarPos("Saturation min", window_name, color_min[1])
    cv.setTrackbarPos("Value max", window_name, color_max[2])
    cv.setTrackbarPos("Value min", window_name, color_min[2])

    while True:

        frame_hsv = cv.cvtColor(src, cv.COLOR_BGR2HSV)
        frame_threshold = cv.inRange(frame_hsv, color_min, color_max)

        frame_threshold = cv.cvtColor(frame_threshold, cv.COLOR_GRAY2RGB)
        img = concat_tile_resize([[src, frame_threshold]])
        cv.imshow(window_name,cv.resize(img, (int(img.shape[1]/2), int(img.shape[0]/2))))

        key = cv.waitKey(30)
        if key == ord('q') or key == 27:
            break
        elif key == ord('n'):
            image_index += 1
            src = cv.imread(dir + "/" + images[image_index])
            src = cv.rotate(src, cv.cv2.ROTATE_90_COUNTERCLOCKWISE)
            print("image ", image_index, '/', len(images))
        elif key == ord('p'):
            image_index -= 1
            src = cv.imread(dir + "/" + images[image_index])
            src = cv.rotate(src, cv.cv2.ROTATE_90_COUNTERCLOCKWISE)
            print("image ", image_index, '/', len(images))
        elif key == ord('s'):
            cv.imwrite(save_dir + '/' + images[image_index], src)
            print("image saved in", save_dir + '/' + images[image_index])

## [main]
if __name__ == "__main__":

    main()