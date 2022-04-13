import numpy as np
class Processing():
    def __init__(self):
        pass

    @staticmethod
    def bounding_box_to_contour(boxes):
        contours = []
        for i, box in enumerate(boxes):
            c = [
                [[box[0], box[1]]],
                [[box[0]+box[2], box[1]]],
                [[box[0]+box[2], box[1]+box[3]]],
                [[box[0], box[1]+box[3]]],
            ]
            contours.append(c)
        a = np.array(contours)
        return a