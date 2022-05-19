import cv2
import numpy as np
import time
from Vision import Vision
from helpers.Filters import Filters

from helpers.PyTorchMLP import PyTorchMLP

from plateFinders.CascadeFinder import CascadeFinder
from plateFinders.RegionFinder import RegionFinder
from plateFinders.ContourFinder import ContourFinder
from plateFinders.mlModel import MlModel
from ocr.OpticalCharacterRecognition import OpticalCharacterRecognition
import torch


def intersection(a,b):
    x = max(a[0], b[0])
    y = max(a[1], b[1])
    w = min(a[0]+a[2], b[0]+b[2]) - x
    h = min(a[1]+a[3], b[1]+b[3]) - y
    if w<0 or h<0: return () # or (0,0,0,0) ?
    return (x, y, w, h)

def test(model, dirname, names):
    ocr = OpticalCharacterRecognition()

    plate_found = 0
    found_field = 0
    images_checked = 0
    for i, name in enumerate(names):
        image_path = "/".join([dirname, name[0]])

        image = cv2.imread(image_path)
        rois = model.predict_rois(image)
        display = Vision.rects(image, rois, tickness=8)

        plates_count = int(name[1])
        rectangles = []

        for j in range(plates_count):
            r = [int(s) for s in name[j + 2:j + 6]]
            r_field = r[2] * r[3]
            rectangles.append(r)
            for k,_ in enumerate(rois):
                inter = intersection(r, rois[k])
                if len(inter) == 4:
                    field = rois[k][2] * rois[k][3]
                    #intersection detected
                    if (field/r_field) < 3:
                        #found!
                        found_field += (field/r_field)
                        print("plate found r =",(field/r_field))
                        plate_found += 1

        images_checked += 1

        ocr.predict(Filters.extract_rois(image, rois))

        cv2.imshow("image", display)
        key = cv2.waitKey(0)

        if key == 27:
           break

    print("Found", plate_found, "on", images_checked, "=>", int(100 * plate_found/images_checked), "% avg field", int(100 * found_field/plate_found), "%")

def learn(model, dirname, names):

    for i, name in enumerate(names):
        image_path = "/".join([dirname, name[0]])
        plates_count = int(name[1])
        rectangles = []
        for j in range(plates_count):
            rectangles.append([int(s) for s in name[j+2:j+6]])

        image = cv2.imread(image_path)
        model.partial_fit(image, rectangles)

        dis = []
        for j, name_ in enumerate(names[0::5]):
            image_path = "/".join([dirname, name_[0]])
            image = cv2.imread(image_path)
            prediction = model.predict(image)
            prediction = cv2.GaussianBlur(prediction, (81,81),0)

            f_pred = prediction.flatten()
            #print("before norm:", f_pred.min(), f_pred.max())
            prediction = ((255 / f_pred.flatten().max()) * prediction).astype(np.uint8)
            #print("after norm:", prediction.min(), prediction.max())

            rects = Filters.object_detection(prediction)
            display = Vision.hconcat_resize_min([cv2.cvtColor(prediction, cv2.COLOR_GRAY2RGB), Vision.rects(image, rects, tickness=8)])
            display = cv2.resize(display, (int(display.shape[1] / 4),  int(display.shape[1] / 4)))
            dis.append(display)

        step = 5
        for k in range(int(len(dis)/step) + 1):
            cv2.imshow("image" + str(k), Vision.hconcat_resize_min(dis[k*step:(k+1)*step]))

        key = cv2.waitKey(100)


        if key == 27:
            break


        print("Progress ", i, "/", len(names))
    key = cv2.waitKey(0)
    #
    # fullness /= len(names)
    # precision /= len(names)
    # timer /= len(names)
    #
    # for i, evaluator in enumerate(evaluators):
    #     print("avg: fullness {0:.3f}, precision {1:.3f}, time: {2:.2f}s, fps: {3:.0f}".format(fullness[i], precision[i], timer[i],
    #                                                                                 1.0 / timer[i]))

    #print(evaluators[0].export_features("asd"))

## [main]
if __name__ == "__main__":

    models = [
        "mlp_models/dobry model 9x9-NN-800-800-800-800-800-800 LeakyReLU-LeakyReLU-LeakyReLU-LeakyReLU-LeakyReLUout: LeakyReLU.pytorch",
        "mlp_models/dobry model 4 9x9NN-800-800-800-800-800-800 LeakyReLU-LeakyReLU-LeakyReLU-LeakyReLU-LeakyReLUout: LeakyReLU.pytorch",
        "mlp_models/dobry model 3 9x9 NN-800-800-800-800-800-800 LeakyReLU-LeakyReLU-LeakyReLU-LeakyReLU-LeakyReLUout: LeakyReLU.pytorch",
        "mlp_models/dobry model 2 9x9 mało czuly.pytorch",
    ]

    model = MlModel(PyTorchMLP(
        path=models[1],
        hidden_layer_sizes=[800, 800, 800, 800, 800, 800],
        hidden_layer_activation=[torch.nn.LeakyReLU(), torch.nn.LeakyReLU(), torch.nn.LeakyReLU(), torch.nn.LeakyReLU(),
                                 torch.nn.LeakyReLU()],
        output_layer_activation=torch.nn.LeakyReLU(),
        tol=1e-8,
        learning_rate=1e-6,
        verbose=True, max_iter=500), verbose=True, image_verbose=True)

    dirname = "positive"
    positive_file = "positive.txt"
    lines = []
    with open("/".join([dirname, positive_file])) as file:
        lines = file.readlines()

    names = [(str.replace('\n', '')).split(' ') for str in lines]
    names = names[4:]

    test(model,dirname, names)
    #learn(model,dirname, names)