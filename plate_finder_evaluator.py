import cv2 as cv
import numpy as np
import time
from Vision import Vision

from plateFinders.CascadeFinder import CascadeFinder
from plateFinders.RegionFinder import RegionFinder
from plateFinders.ContourFinder import ContourFinder

def intersection(a,b):
    x = max(a[0], b[0])
    y = max(a[1], b[1])
    w = min(a[0]+a[2], b[0]+b[2]) - x
    h = min(a[1]+a[3], b[1]+b[3]) - y
    if w<0 or h<0: return () # or (0,0,0,0) ?
    return (x, y, w, h)

def area(rect):
    return rect[3]*rect[2]



def fit_finder(image_path, rectangles, evaluators, iterator):

    #found rects area, intersections area with plates, computation time
    stats = np.zeros([len(evaluators), 3])
    plates_area = 0

    colors = [(0,255,0),(0,255,255),(255,0,255),(255,255,0),(0,0,0),(0,0,255)]

    image = cv.imread(image_path)
    print(image_path)
    image = image[600:-100, :]
    output = image.copy()

    for _, rect in enumerate(rectangles):
        cv.rectangle(output, (rect[0], rect[1]), (rect[0]+rect[2], rect[1]+rect[3]), (255,0,0), 1)
        plates_area += area(rect)

    counter = 0
    for i, evaluator in enumerate(evaluators):
        start = time.time()
        contours = evaluator.fit(image, rectangles)
        stats[i, 2] = time.time() - start

        #steps = []
        #for _, image in enumerate(evaluator.image_processing_steps_):
        #    #steps.append(cv.resize(image, (int(image.shape[1]/2), int(image.shape[0]/2))))
        #    steps.append(image)

        #cv.imshow("Steps", Vision.hconcat_resize_min(evaluator.image_processing_steps_))
        #cv.imshow("Steps for e: " + str(i), Vision.hconcat_resize_min(steps))


        for _, contour in enumerate(contours):
            rect = cv.boundingRect(contour)
            x, y, w, h = rect
            cv.drawContours(output, [contour], 0, colors[i%len(colors)], 2)
            cv.rectangle(output, (x, y), (x + w, y + h), colors[i%len(colors) +1 ], 2)
            #cv.imwrite("plates2/img_" + str(counter) + "_" + str(iterator) + ".png", image[y:y + h, x:x + w])
            #cv.imshow("plates2/img_" + str(counter) + "_" + str(iterator) + ".png", image[y:y + h, x:x + w])
            counter += 1
            stats[i,0] += area(rect)

            #measure statistics
            for _, plate in enumerate(rectangles):
                inter = intersection(rect, plate)
                if(len(inter) > 0):
                    stats[i,1] += area(inter)
                    cv.rectangle(output, (inter[0],inter[1]), (inter[0] + inter[2], inter[1] + inter[3]), (252,157,3), 1)

            #cv.rectangle(image, (rect[0], rect[1]), (rect[0] + rect[2], rect[1] + rect[3]), (255, 0, 0), 3)

    fullness = np.zeros(len(evaluators))
    precision = np.zeros(len(evaluators))
    timer = stats[:, 2]
    for i, evaluator in enumerate(evaluators):
        if plates_area > 0 and stats[i,0] > 0:
            fullness[i] = stats[i,1] / plates_area
            precision[i] = plates_area / stats[i,0]

        print("fullness {0:.3f},precision {1:.3f}, time: {2:.2f}s, fps: {3:.0f}".format(fullness[i], precision[i], timer[i], 1.0/timer[i]))

    cv.imshow("image", output)
    return  fullness, precision, timer


def image_process(image_path, rectangles, evaluators):

    #found rects area, intersections area with plates, computation time
    stats = np.zeros([len(evaluators), 3])
    plates_area = 0

    colors = [(0,255,0),(0,255,255),(255,0,255),(255,255,0),(0,0,0),(0,0,255)]

    image = cv.imread(image_path)
    output = image.copy()
    counter = 0
    for _, rect in enumerate(rectangles):
        cv.rectangle(output, (rect[0], rect[1]), (rect[0]+rect[2], rect[1]+rect[3]), (255,0,0), 2)
        plates_area += area(rect)

    for i, evaluator in enumerate(evaluators):
        start = time.time()
        rects = evaluator.detectMultiScale(image)
        stats[i, 2] = time.time() - start

        for _, rect in enumerate(rects):
            x, y, w, h = rect

            cv.rectangle(output, (x, y), (x + w, y + h), colors[i%len(colors)], 2)
            stats[i,0] += area(rect)

            #measure statistics
            for _, plate in enumerate(rectangles):
                inter = intersection(rect, plate)
                if(len(inter) > 0):
                    stats[i,1] += area(inter)
                    cv.rectangle(output, (inter[0],inter[1]), (inter[0] + inter[2], inter[1] + inter[3]), (252,157,3), 2)

            #cv.rectangle(image, (rect[0], rect[1]), (rect[0] + rect[2], rect[1] + rect[3]), (255, 0, 0), 2)

    fullness = np.zeros(len(evaluators))
    precision = np.zeros(len(evaluators))
    timer = stats[:, 2]
    for i, evaluator in enumerate(evaluators):
        if plates_area > 0 and stats[i,0] > 0:
            fullness[i] = stats[i,1] / plates_area
            precision[i] = plates_area / stats[i,0]

        print("fullness {0:.3f},precision {1:.3f}, time: {2:.2f}s, fps: {3:.0f}".format(fullness[i], precision[i], timer[i], 1.0/timer[i]))

    cv.imshow("image", output)
    return  fullness, precision, timer




def main():

    #define finder
    evaluators = [
    #    CascadeFinder("haar_cascades/cascade1.xml"),
    #    CascadeFinder("haar_cascades/cascade48x24.xml")
        RegionFinder()
    #    ContourFinder()
    ]


    fullness, precision, timer = [0,0,0]

    dirname = "positive"
    positive_file = "positive.txt"
    lines = []
    with open("/". join([dirname, positive_file])) as file:
        lines = file.readlines()

    names = [(str.replace('\n', '')).split(' ') for str in lines]
    print(names)
    for i, name in enumerate(names):
        image_path = "/".join([dirname, name[0]])
        plates_count = int(name[1])
        rectangles = []
        for j in range(plates_count):
            rectangles.append([int(s) for s in name[j+2:j+6]])

        f,p,t = fit_finder(image_path, rectangles, evaluators, i)
        #f,p,t = image_process(image_path, rectangles, evaluators)
        fullness += f
        precision += p
        timer += t
        key = cv.waitKey(1)

        if key == 27:
            break

        print("Progress ", i, "/", len(names))

    fullness /= len(names)
    precision /= len(names)
    timer /= len(names)

    for i, evaluator in enumerate(evaluators):
        print("avg: fullness {0:.3f}, precision {1:.3f}, time: {2:.2f}s, fps: {3:.0f}".format(fullness[i], precision[i], timer[i],
                                                                                    1.0 / timer[i]))

    #print(evaluators[0].export_features("asd"))

## [main]
if __name__ == "__main__":
    main()