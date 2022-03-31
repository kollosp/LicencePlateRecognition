import cv2 as cv
import numpy as np
import time
from Vision import Vision
from ocr.OpticalCharacterRecognition import OpticalCharacterRecognition

from plateFinders.CascadeFinder import CascadeFinder
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

def image_process(image_path, rectangles, plate_finder, ocr):

    image = cv.imread(image_path)
    image = image[600:-100, :]
    output = image.copy()

    for _, rect in enumerate(rectangles):
        cv.rectangle(output, (rect[0], rect[1]), (rect[0]+rect[2], rect[1]+rect[3]), (255,0,0), 2)

    start = time.time()
    contours = plate_finder.detectMultiScale(image)

    boxes = []
    images = []
    for _, contour in enumerate(contours):
        rect = cv.boundingRect(contour)
        boxes.append(rect)
        x, y, w, h = rect
        images.append(image[y:y+h,x:x+w])

    ocr.predict(images)
    timer = time.time() - start

    for _,rect in enumerate(boxes):
        x, y, w, h = rect
        cv.rectangle(output, (x, y), (x + w, y + h), (0,255,0), 2)
        cv.rectangle(image, (rect[0], rect[1]), (rect[0] + rect[2], rect[1] + rect[3]), (255, 0, 0), 2)

    print("time: {0:.2f}s, fps: {1:.0f}".format(timer, 1.0/timer))

    cv.imshow("image", output)
    return  timer




def main():

    plate_finder = ContourFinder()
    ocr = OpticalCharacterRecognition()
    timer = 0

    dirname = "cascade"
    positive_file = "pos.txt"
    lines = []
    with open("/". join([dirname, positive_file])) as file:
        lines = file.readlines()

    names = [(str.replace('\n', '')).split(' ') for str in lines]
    names = names[18:]
    for i, name in enumerate(names):
        image_path = "/".join([dirname, name[0]])
        plates_count = int(name[1])
        rectangles = []
        for j in range(plates_count):
            rectangles.append([int(s) for s in name[j+2:j+6]])

        t = image_process(image_path, rectangles, plate_finder, ocr)
        timer += t
        key = cv.waitKey(0)

        if key == 27:
            break

        print("Progress ", i, "/", len(names))

    timer /= len(names)
    print("avg: time: {0:.2f}s, fps: {1:.0f}".format(timer,1.0 / timer))

    #print(evaluators[0].export_features("asd"))

## [main]
if __name__ == "__main__":
    main()