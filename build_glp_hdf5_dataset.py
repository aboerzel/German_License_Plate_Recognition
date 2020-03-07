import argparse
import os

import cv2
import numpy as np
import progressbar
from imutils import paths

from config import config
from utils.io import HDF5DatasetWriter

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image_data", default=config.PLATE_IMAGES, help="image data path")
ap.add_argument("-d", "--dataset", default=config.GLP_HDF5, help="glp dataset hdf5-file")
args = vars(ap.parse_args())

# read image paths and labels
paths = list(paths.list_images(args['image_data']))
labels = [p.split(os.path.sep)[-1].split(".")[0].split('#')[1] for p in paths]

# original size of generated license plate images
IMAGE_WIDTH = 151
IMAGE_HEIGHT = 32

# create HDF5 writer
print("[INFO] building {}...".format(args['dataset']))
writer = HDF5DatasetWriter((len(paths), IMAGE_HEIGHT, IMAGE_WIDTH), args['dataset'])

# initialize the progress bar
widgets = ["Building Dataset: ", progressbar.Percentage(), " ", progressbar.Bar(), " ", progressbar.ETA()]
pbar = progressbar.ProgressBar(maxval=len(paths), widgets=widgets)
pbar.start()

# loop over the image paths
for (i, (path, label)) in enumerate(zip(paths, labels)):
    # load the image and process it
    # image = cv2.imread(path, cv2.IMREAD_COLOR)  # don't use imread because bug with utf-8 paths
    stream = open(path, "rb")
    bytes = bytearray(stream.read())
    numpyarray = np.asarray(bytes, dtype=np.uint8)

    try:
        image = cv2.imdecode(numpyarray, cv2.IMREAD_GRAYSCALE)
    except:
        print("open image failed: %s" % path)
        continue

    # check image size
    if not image.shape == (IMAGE_HEIGHT, IMAGE_WIDTH):
        print("image with wrong size: %s" % path)
        continue

    # check number length
    if len(label) > 10:
        print("image with wrong label: %s - %s" % (path, label))
        continue

    # add the image and label # to the HDF5 images
    writer.add([image], [label])
    pbar.update(i)

# close the HDF5 writer
pbar.finish()
writer.close()
