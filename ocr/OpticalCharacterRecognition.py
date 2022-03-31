import cv2
import numpy as np
from Vision import Vision
from scipy import stats
from lib.helpers import rerange
class OpticalCharacterRecognition:
    def __init__(self):
        self.image_processing_steps_ = []

    def predict(self, images):
        for i, image in enumerate(images):
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            th = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 7, -2)
            #dilation_size = 1
            #th = cv2.erode(th,cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * dilation_size + 1, 2 * dilation_size + 1),
            #                                                                 (dilation_size, dilation_size)))

            #print(th.shape)
            #positions = np.array([[j,i] for i in range(len(th)) for j in range(len(th[i])) if th[i,j] > 0])
            #print(positions)
            #kernel = stats.gaussian_kde(np.vstack([positions[:,0].ravel(), positions[:,1].ravel()]))
            #all_positions = np.array([[j, i] for i in range(len(th)) for j in range(len(th[i]))])
            #density = np.reshape(kernel(np.vstack([all_positions[:,0].ravel(), all_positions[:,1].ravel()])),th.shape)
            #density = rerange(density, (0,255))
            #print(density.shape)
            #print(density.min(), density.max())
            gaussian =cv2.blur(th, (9,9))
            #gaussian = cv2.GaussianBlur(th, (9, 9), 0)
            dilation_size = 4
            local_max_limits = cv2.dilate(gaussian,cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * dilation_size + 1, 2 * dilation_size + 1),
                                                                             (dilation_size, dilation_size)))
            dilated = local_max_limits.copy()
            local_max_limits[gaussian >= local_max_limits] = 255
            local_max_limits[local_max_limits != 255] = 0

            ret, markers = cv2.connectedComponents(local_max_limits)

            bgr = cv2.cvtColor(th, cv2.COLOR_GRAY2BGR)
            markers = cv2.watershed(bgr, markers)
            markers = markers + 1
            segments = np.zeros(th.shape, np.uint8)
            areas_image = np.zeros(th.shape, np.uint8)
            areas = []
            markers_count = markers.max()
            for j in range(markers_count):
                step = 255/markers_count
                segments[markers == j] = int(j*step) + step
                mask = np.zeros(th.shape, np.uint8)
                mask[markers == j] = 255
                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                for k, contour in enumerate(contours):
                    bounding_box = cv2.boundingRect(contour)
                    areas.append(bounding_box)
                    Vision.rect(areas_image, bounding_box,int(j*step) + step)
                    print(self.extract_features(gray, bounding_box))

            signs = []
            for j, area in enumerate(areas):
                signs.append(th[area[1]:area[1]+area[3], area[0]:area[0]+area[2]])

            if len(signs) > 0:
                print("signs",len(signs))
                cv2.imshow("signs" + str(i), Vision.hconcat_resize_min(signs))
                cv2.imshow("plate" + str(i), Vision.resize(Vision.vconcat_resize_min([gray, th, gaussian, local_max_limits,areas_image, segments]),2))

    def extract_features(self, image, box, f_grid=(5,5)):
        f_matrix = np.zeros(f_grid)
        roi = image[box[1]:box[1]+box[3], box[0]:box[0]+box[2]]
        fields = f_grid[0] * f_grid[1] * 255
        for i in range(f_grid[0]):
            x_d = roi.shape[1] / f_grid[0]
            for j in range(f_grid[1]):
                y_d = roi.shape[0] / f_grid[1]
                f_matrix[i,j] = roi[int(j*y_d):int((j+1)*y_d), int(i*x_d):int((i+1)*x_d)].flatten().sum() / fields

        return f_matrix

    def fit(self, images, texts):
        pass