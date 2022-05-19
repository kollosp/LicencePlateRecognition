import cv2 as cv
import numpy as np

class Vision:
    def __init__(self):
        pass


    @staticmethod
    def rects(image, rects, color=(255,0,0), tickness=1):
        if rects is None:
            return image

        img = image.copy()
        for i, rect in enumerate(rects):
            cv.rectangle(img, (rect[0], rect[1]), (rect[0]+rect[2], rect[1]+rect[3]),color, tickness)
        return img

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
    def intersperse(lst, item):
        result = [item] * (len(lst) * 2 - 1)
        result[0::2] = lst
        return result

    @staticmethod
    def hconcat_resize_min(im_list, interpolation=cv.INTER_CUBIC, borders_thickness=0):
        h_min = max(im.shape[0] for im in im_list)

        if borders_thickness > 0:
            splitter = np.ones((borders_thickness, 1), np.uint8)
            splitter.fill(255)

            im_list = Vision.intersperse(im_list, splitter)

            #print("borders_thickness", borders_thickness)
            #length = len(im_list)
            #for i in range(length-1, 1, 1):
            #    print(i)
            #    im_list.insert(i, np.array((borders_thickness, h_min)).fill(255))

        im_list_resize = [cv.resize(im, (int(im.shape[1] * h_min / im.shape[0]), h_min), interpolation=interpolation)
                          for im in im_list if (im.shape[1] * h_min / im.shape[0]) > 1]
        return cv.hconcat(im_list_resize)

    @staticmethod
    def concat_tile_resize(im_list_2d, interpolation=cv.INTER_CUBIC, borders_thickness=0):
        im_list_v = [Vision.hconcat_resize_min(im_list_h, interpolation=interpolation,borders_thickness=borders_thickness) for im_list_h in im_list_2d]
        return Vision.vconcat_resize_min(im_list_v, interpolation=interpolation, borders_thickness=borders_thickness)
