import argparse
import tarfile

import cv2
import numpy
import progressbar
import numpy as np

from config import config
from utils.io import HDF5DatasetWriter

IMAGE_WIDTH = 256
IMAGE_HEIGHT = 256

ap = argparse.ArgumentParser()
ap.add_argument("-s", "--source", default=config.SUN397_TAR_FILE, help="source tar-file")
ap.add_argument("-t", "--target", default=config.BACKGRND_HDF5, help="target output hdf5-file")
ap.add_argument("-i", "--items", default=1000, type=int, help="max images")
args = vars(ap.parse_args())


def im_from_file(f):
    a = numpy.asarray(bytearray(f.read()), dtype=numpy.uint8)
    return cv2.imdecode(a, cv2.IMREAD_GRAYSCALE)


def extract_backgrounds(archive_name, output_path, max_items=np.inf):
    print("[INFO] reading content of {}...".format(archive_name))
    tar = tarfile.open(name=archive_name)
    files = tar.getnames()

    # create shuffled index list
    randomized_indexes = np.arange(len(files))
    np.random.shuffle(randomized_indexes)

    # pick max number of items
    if max_items == np.inf or max_items > len(files):
        max_items = len(files)

    randomized_indexes = randomized_indexes[0:max_items]

    print("[INFO] building {}...".format(output_path))
    writer = HDF5DatasetWriter((len(randomized_indexes), IMAGE_HEIGHT, IMAGE_WIDTH), output_path)

    widgets = ["Building Dataset: ", progressbar.Percentage(), " ", progressbar.Bar(), " ", progressbar.ETA()]
    pbar = progressbar.ProgressBar(maxval=len(randomized_indexes), widgets=widgets).start()
    index = 0

    for i, file in enumerate(files):

        if i not in randomized_indexes:
            continue

        f = tar.extractfile(file)
        if f is None:
            continue  # skip directories
        try:
            image = im_from_file(f)
        finally:
            f.close()
        if image is None:
            continue  # skip non image files

        # make same width and height, by cutting the larger dimension to the smaller dimension
        if image.shape[0] > image.shape[1]:
            image = image[:image.shape[1], :]
        else:
            image = image[:, :image.shape[0]]

        # resize to target-width and -height, keeping the aspect ratio
        if image.shape[0] != 256:
            image = cv2.resize(image, (256, 256))

        # name from index
        name = "{:08}".format(index)

        # check image size
        if not image.shape == (IMAGE_HEIGHT, IMAGE_WIDTH):
            print("image with wrong size: %s" % name)
            continue

        # add the image and name to the HDF5 db
        writer.add([image], [name])
        pbar.update(index)
        index += 1

    # close the HDF5 writer
    pbar.finish()
    writer.close()
    print("[INFO] {} images saved to {}...".format(len(randomized_indexes), output_path))


if __name__ == "__main__":
    max_items = np.inf
    if args['items'] > 0:
        max_items = args['items']
    extract_backgrounds(args['source'], args['target'], max_items)
