# import the necessary packages
from PIL import Image
import numpy as np


class AspectAwarePreprocessor:
    def __init__(self, width, height):
        self.width = width
        self.height = height

    def preprocess(self, image):
        ratio = float(self.width) / image.size[0]
        new_size = tuple([int(x * ratio) for x in image.size])
        image = image.resize(new_size, Image.LANCZOS)  # LANCZOS, ANTIALIAS, BILINEAR, BICUBIC, NEAREST
        # create a new image and paste the resized on it
        new_im = Image.new("F", (self.width, self.height))
        y = (self.height - new_size[1]) // 2
        new_im.paste(image, (0, y))
        return np.array(new_im)
