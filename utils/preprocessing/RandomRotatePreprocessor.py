import cv2
import random
import numpy as np


class RandomRotatePreprocessor:
    def __init__(self, min_angle, max_angle, width, height, inter=cv2.INTER_AREA):
        self.min = min_angle
        self.max = max_angle
        self.width = width
        self.height = height
        self.inter = inter

    def preprocess(self, image):
        angle = random.randint(self.min, self.max)
        rotated = self.rotate_bound(image, angle)
        return cv2.resize(rotated, (self.width, self.height), interpolation=self.inter)

    def rotate_bound(self, image, angle):
        # grab the dimensions of the image and then determine the
        # center
        (h, w) = image.shape[:2]
        (cX, cY) = (w // 2, h // 2)

        # grab the rotation matrix (applying the negative of the
        # angle to rotate clockwise), then grab the sine and cosine
        # (i.e., the rotation components of the matrix)
        M = cv2.getRotationMatrix2D((cX, cY), -angle, 1.0)
        cos = np.abs(M[0, 0])
        sin = np.abs(M[0, 1])

        # compute the new bounding dimensions of the image
        nW = int((h * sin) + (w * cos))
        nH = int((h * cos) + (w * sin))

        # adjust the rotation matrix to take into account translation
        M[0, 2] += (nW / 2) - cX
        M[1, 2] += (nH / 2) - cY

        # perform the actual rotation and return the image
        # return cv2.warpAffine(image, M, (nW, nH), borderMode=cv2.BORDER_TRANSPARENT)
        return cv2.warpAffine(image, M, (nW, nH))
