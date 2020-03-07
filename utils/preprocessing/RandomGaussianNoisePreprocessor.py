import numpy as np
import random


class RandomGaussianNoisePreprocessor:
    def __init__(self, noise_sigma):
        self.noise_sigma = noise_sigma

    def preprocess(self, image):
        temp_image = np.float64(np.copy(image))

        h = temp_image.shape[0]
        w = temp_image.shape[1]
        noise = np.random.randn(h, w) * random.randint(0, self.noise_sigma)

        noisy_image = np.zeros(temp_image.shape, np.float64)
        if len(temp_image.shape) == 2:
            noisy_image = temp_image + noise
        else:
            noisy_image[:, :, 0] = temp_image[:, :, 0] + noise
            noisy_image[:, :, 1] = temp_image[:, :, 1] + noise
            noisy_image[:, :, 2] = temp_image[:, :, 2] + noise

        return noisy_image
