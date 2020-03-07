# import the necessary packages
import os

# root paths
DATASET_ROOT_PATH = os.path.join(os.getcwd(), "..", "datasets", "german_license_plates")
SUN397_TAR_FILE = os.path.join(os.getcwd(), "..", "datasets", "SUN397.tar.gz")

# network image size
IMAGE_WIDTH = 128
IMAGE_HEIGHT = 64

# training parameter
BATCH_SIZE = 64
NUM_EPOCHS = 1000

# license number construction
DOWNSAMPLE_FACTOR = 2 ** 2  # <= pool size ** number of pool layers
MAX_TEXT_LEN = 10

# model name
MODEL_NAME = "glpr-model"

# supported optimizer methods: sdg, rmsprop, adam, adagrad, adadelta
OPTIMIZER = "adagrad"

# define the path to the output directory used for storing plots, classification reports, etc.
OUTPUT_PATH = "output"
DOCUMENTATION_PATH = "documentation"

# json file with the list of german county marks
GERMAN_COUNTY_MARKS = "./config/german_county_marks.json"

# define the paths to the training and validation directories
PLATE_IMAGES = os.path.join(DATASET_ROOT_PATH, "images")
TEST_IMAGES = "testimages"

# define the path to the output training, validation, and testing HDF5 files
GLP_HDF5 = os.path.join(DATASET_ROOT_PATH, "glp.h5")
BACKGRND_HDF5 = os.path.join(DATASET_ROOT_PATH, "background.h5")
