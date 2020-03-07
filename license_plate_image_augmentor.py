import math
import random

import cv2
import numpy as np


class LicensePlateImageAugmentor:
    def __init__(self, img_w, img_h, background_images):

        self.OUTPUT_SHAPE = img_h, img_w
        self.background_images, _ = background_images

    def __get_random_background_image__(self):
        index = random.randint(0, len(self.background_images) - 1)
        return self.background_images[index]

    def __generate_background_image__(self):
        background = self.__get_random_background_image__()
        x = random.randint(0, background.shape[1] - self.OUTPUT_SHAPE[1])
        y = random.randint(0, background.shape[0] - self.OUTPUT_SHAPE[0])
        background = background[y:y + self.OUTPUT_SHAPE[0], x:x + self.OUTPUT_SHAPE[1]]
        return background

    @staticmethod
    def __euler_to_mat__(yaw, pitch, roll):
        # Rotate clockwise about the Y-axis
        c, s = math.cos(yaw), math.sin(yaw)
        M = np.matrix([[c, 0., s],
                       [0., 1., 0.],
                       [-s, 0., c]])

        # Rotate clockwise about the X-axis
        c, s = math.cos(pitch), math.sin(pitch)
        M = np.matrix([[1., 0., 0.],
                       [0., c, -s],
                       [0., s, c]]) * M

        # Rotate clockwise about the Z-axis
        c, s = math.cos(roll), math.sin(roll)
        M = np.matrix([[c, -s, 0.],
                       [s, c, 0.],
                       [0., 0., 1.]]) * M

        return M

    def __make_affine_transform__(self, from_shape, to_shape, rotation_variation=1.0):

        from_size = np.array([[from_shape[1], from_shape[0]]]).T
        to_size = np.array([[to_shape[1], to_shape[0]]]).T

        roll = random.uniform(-0.3, 0.3) * rotation_variation
        pitch = random.uniform(-0.2, 0.2) * rotation_variation
        yaw = random.uniform(-1.2, 1.2) * rotation_variation

        scale = 0.8

        center_to = to_size / 2.
        center_from = from_size / 2.

        M = self.__euler_to_mat__(yaw, pitch, roll)[:2, :2]
        M *= scale
        M = np.hstack([M, center_to - M * center_from])

        return M

    @staticmethod
    def __gaussian_noise__(image, sigma=1):
        mean = 0.0
        gauss = np.random.normal(mean, sigma, image.shape)
        image = image + gauss
        return image

    @staticmethod
    def __brightness__(img, factor=0.5):
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)  # convert to hsv
        hsv = np.array(hsv, dtype=np.float64)
        hsv[:, :, 2] = hsv[:, :, 2] * (factor + np.random.uniform())  # scale channel V uniformly
        hsv[:, :, 2][hsv[:, :, 2] > 255] = 255  # reset out of range values
        rgb = cv2.cvtColor(np.array(hsv, dtype=np.uint8), cv2.COLOR_HSV2RGB)
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)

    @staticmethod
    def __blur__(img):
        blur_value = random.randint(0, 2) + 1
        img = cv2.blur(img, (blur_value, blur_value))
        return img

    @staticmethod
    def __normalize_image__(image):
        # normalize image data between 0 and 1
        # image = (image - image.min()) / (image.max() - image.min())
        image = image.astype(np.float32) / 255.
        return image

    def generate_plate_image(self, plate_img):
        bi = self.__generate_background_image__()

        random_brightness = random.uniform(0.0, 0.7)
        bi = self.__brightness__(bi, random_brightness)
        plate_img = self.__brightness__(plate_img, random_brightness)

        M = self.__make_affine_transform__(
            from_shape=plate_img.shape,
            to_shape=bi.shape,
            rotation_variation=0.8)

        plate_mask = np.ones(plate_img.shape)
        plate_img = cv2.warpAffine(plate_img, M, (bi.shape[1], bi.shape[0]))
        plate_mask = cv2.warpAffine(plate_mask, M, (bi.shape[1], bi.shape[0]))

        out = plate_img * plate_mask + bi * (1.0 - plate_mask)
        #out = self.__gaussian_noise__(out, random.randrange(1, 10))
        out = self.__blur__(out)
        out = self.__normalize_image__(out)
        return out
