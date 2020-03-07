import argparse
import os
import random

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from tensorflow_core.lite.python.interpreter import Interpreter

from config import config
from label_codec import LabelCodec
from utils.preprocessing import AspectAwarePreprocessor

ap = argparse.ArgumentParser()
ap.add_argument("-o", "--optimizer", default=config.OPTIMIZER,
                help="supported optimizer methods: sdg, rmsprop, adam, adagrad, adadelta")
args = vars(ap.parse_args())

OPTIMIZER = args["optimizer"]
MODEL_PATH = os.path.join(config.OUTPUT_PATH, OPTIMIZER, config.MODEL_NAME) + ".h5"
SAVED_MODEL_PATH = os.path.join(config.OUTPUT_PATH, OPTIMIZER, "saved_model")
TFLITE_MODEL_PATH = os.path.join(config.OUTPUT_PATH, OPTIMIZER, config.MODEL_NAME) + ".tflite"

print("Optimizer:    {}".format(OPTIMIZER))
print("Model path:   {}".format(MODEL_PATH))

img_filename = random.choice(os.listdir(config.TEST_IMAGES))
img_filepath = os.path.join(config.TEST_IMAGES, img_filename)
label = img_filename.split(".")[0].split("#")[0]

p = AspectAwarePreprocessor(config.IMAGE_WIDTH, config.IMAGE_HEIGHT)

image = Image.open(img_filepath)
image = p.preprocess(image)
image = image.astype(np.float32) / 255.
image = np.expand_dims(image.T, axis=-1)

# Load TFLite model and allocate tensors.
interpreter = Interpreter(model_path=TFLITE_MODEL_PATH)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

input_shape = input_details[0]['shape']
output_shape = output_details[0]['shape']
print("Input Shape:  {}".format(input_shape))
print("Input Details: {}".format(input_details))
print("Output Details: {}".format(output_details))

interpreter.set_tensor(input_details[0]['index'], np.asarray([image]))
interpreter.invoke()
predictions = interpreter.get_tensor(output_details[0]['index'])

label = LabelCodec.decode_prediction(predictions[0])


plt.axis("off")
plt.title(label)
plt.imshow(image[:, :, 0].T, cmap='gray')
plt.show()

