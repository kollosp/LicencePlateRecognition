import cv2
import numpy as np
from Vision import Vision
from helpers.Filters import Filters

from helpers.PyTorchMLP import PyTorchMLP

from plateFinders.mlModel import MlModel
from signFinders.SkeletonSearch import SkeletonSearch
from sklearn.model_selection import KFold
from sklearn.linear_model import SGDClassifier

from ocr.mlOCR import MlOCR
import torch

def main():
    size = (13,13)
    shift = 1
    ocr = MlOCR(SGDClassifier(loss="modified_huber"), size, path="ocr_models/1.bin")
    #ocr = MlOCR.load("ocr_models/1.bin")
    file = open("dataset_letters/img_letters.txt")
    in_file = file.read()
    dataset = in_file.split('\n')
    dataset = [l.split(',') for l in dataset]
    file.close()

    X=[]
    labels=[]


    for i in range(len(dataset)-1):
        X.append(dataset[i][0])
        for j in range(6):
            labels.append(dataset[i][1])

    images =[]

    #learning set extension
    for i in range(len(X)):
        image = cv2.imread(f"dataset_letters/{X[i]}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = cv2.adaptiveThreshold(image,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV,11,2)
        #cv2.imshow("img", image)
        #cv2.waitKey(100)
        images.append(image)

        image_tmp = MlOCR.shift_image(np.copy(image),0,-shift)
        images.append(image_tmp)
        image_tmp = MlOCR.shift_image(np.copy(image),0,shift)
        images.append(image_tmp)
        image_tmp = MlOCR.shift_image(np.copy(image),-shift, 0)
        images.append(image_tmp)
        image_tmp = MlOCR.shift_image(np.copy(image),shift,0)
        images.append(image_tmp)
        image_tmp = cv2.blur(image, (3,3))
        images.append(image_tmp)

    images = np.array(images)
    labels = np.array(labels)
    ocr.fit(images, labels)
    ocr.save("ocr_models/1.bin")

## [main]
if __name__ == "__main__":
    main()