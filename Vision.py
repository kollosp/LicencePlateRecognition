import cv2 as cv
import numpy as np

class Vision:
    def __init__(self):
        pass

    @staticmethod
    def rect(image, rect, color=(255,0,0), tickness=1):
        cv.rectangle(image, (rect[0], rect[1]), (rect[0]+rect[2], rect[1]+rect[3]),color, tickness)

    @staticmethod
    def resize(image, factor, interpolation=cv.INTER_CUBIC):
        return cv.resize(image, (int(image.shape[0]* factor),int(image.shape[1]* factor)), interpolation=interpolation)

    @staticmethod
    def vconcat_resize_min(im_list, interpolation=cv.INTER_CUBIC, borders_thickness=0):
        w_min = min(im.shape[1] for im in im_list)
        im_list_resize = [cv.resize(im, (w_min, int(im.shape[0] * w_min / im.shape[1])), interpolation=interpolation)
                          for im in im_list]
        return cv.vconcat(im_list_resize)

    @staticmethod
    def hconcat_resize_min(im_list, interpolation=cv.INTER_CUBIC, borders_thickness=0):
        h_min = min(im.shape[0] for im in im_list)

        if borders_thickness > 0:
            print("borders_thickness", borders_thickness)
            length = len(im_list)
            for i in range(length-1, 1, 1):
                print(i)
                im_list.insert(i, np.array((borders_thickness, h_min)).fill(255))
        print(h_min)
        print([(int(1.0 * im.shape[1] * h_min / im.shape[0]), h_min) for im in im_list])
        im_list_resize = [cv.resize(im, (int(im.shape[1] * h_min / im.shape[0]), h_min), interpolation=interpolation)
                          for im in im_list]
        return cv.hconcat(im_list_resize)

    @staticmethod
    def concat_tile_resize(im_list_2d, interpolation=cv.INTER_CUBIC, borders_thickness=0):
        im_list_v = [Vision.hconcat_resize_min(im_list_h, interpolation=interpolation,borders_thickness=borders_thickness) for im_list_h in im_list_2d]
        return Vision.vconcat_resize_min(im_list_v, interpolation=interpolation, borders_thickness=borders_thickness)
