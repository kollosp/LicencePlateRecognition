import numpy as np

def rerange(image, expected):
    max = np.max(image)
    min = np.min(image)
    delta = max - min

    image = 1.0 * (image - min) / delta * expected[1]  + expected[0]

    return image.astype(np.uint8)