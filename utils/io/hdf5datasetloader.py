import h5py
import numpy as np


class Hdf5DatasetLoader:
    def __init__(self, preprocessors=None):
        self.preprocessors = preprocessors

        # if the preprocessors are None, initialize them as an empty list
        if self.preprocessors is None:
            self.preprocessors = []

    def load(self, db_path, shuffle=False, max_items=np.inf):

        db = h5py.File(db_path, 'r')
        images = np.array(db["images"])
        labels = np.array(db["labels"])
        db.close()

        if shuffle:
            randomized_indexes = np.arange(len(images))
            np.random.shuffle(randomized_indexes)
            images = images[randomized_indexes]
            labels = labels[randomized_indexes]

        if max_items == np.inf or max_items > len(images):
            max_items = len(images)

        images = images[0:max_items]
        labels = labels[0:max_items]

        # preprocess images
        for i, (image, label) in enumerate(zip(images, labels)):

            for p in self.preprocessors:
                image = p.preprocess(image)
                images[i] = image

        return images, labels
