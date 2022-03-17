import cv2 as cv

class CascadeFinder:
    def __init__(self, file=None):
        self.file = file
        self.cascade = cv.CascadeClassifier()

        if self.file is not None:
            self.load_from_file(self.file)
        pass

    def load_from_file(self, file):
        self.file = file
        self.cascade.load(file)

    def detectMultiScale(self, image):
        if self.cascade is None:
            print("You have to fit this object first. Use load_from_file function")

        return self.cascade.detectMultiScale(image)