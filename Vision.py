import cv2 as cv


class Vision:
    def __init__(self):
        pass

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
        im_list_v = [Vision.hconcat_resize_min(im_list_h, interpolation=interpolation) for im_list_h in im_list_2d]
        return Vision.vconcat_resize_min(im_list_v, interpolation=interpolation)
