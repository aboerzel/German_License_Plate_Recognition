import argparse
import os
import numpy as np
import tensorflow as tf

from tensorflow_core.lite.python.interpreter import Interpreter
from tensorflow_core.lite.python.lite import TFLiteConverter, Optimize

from config.license_recognition import config

ap = argparse.ArgumentParser()
ap.add_argument("-o", "--optimizer", default=config.OPTIMIZER, help="supported optimizer methods: sdg, rmsprop, adam, adagrad, adadelta")
args = vars(ap.parse_args())

OPTIMIZER = args["optimizer"]
MODEL_PATH = os.path.join(config.OUTPUT_PATH, OPTIMIZER, config.MODEL_NAME) + ".h5"
SAVED_MODEL_PATH = os.path.join(config.OUTPUT_PATH, OPTIMIZER, "saved_model")
TFLITE_MODEL_PATH = os.path.join(config.OUTPUT_PATH, OPTIMIZER, config.MODEL_NAME) + "-float.tflite"

print("Optimizer:    {}".format(OPTIMIZER))
print("Model path:   {}".format(MODEL_PATH))

converter = TFLiteConverter.from_keras_model_file(MODEL_PATH)
# converter = TFLiteConverter.from_saved_model(SAVED_MODEL_PATH)
# converter = TFLiteConverter.from_keras_model(model)
converter.optimizations.append(Optimize.DEFAULT)
converter.target_spec.supported_types.append(tf.float16)
converter.experimental_new_converter = True
# converter.post_training_quantize = True
tflite_model = converter.convert()

open(TFLITE_MODEL_PATH, "wb").write(tflite_model)


# Load TFLite model and allocate tensors.
interpreter = Interpreter(model_path=TFLITE_MODEL_PATH)
interpreter.allocate_tensors()
# Get input and output tensors
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
# Test model on random input data
input_shape = input_details[0]['shape']
output_shape = output_details[0]['shape']
print("Input Shape:  {}".format(input_shape))
print("Input Details: {}".format(input_details))
print("Output Details: {}".format(output_details))
input_data = np.array(np.random.random_sample(input_shape), dtype=np.float32)
interpreter.set_tensor(input_details[0]['index'], input_data)
interpreter.invoke()
output_data = interpreter.get_tensor(output_details[0]['index'])
print("Output: {}".format(output_data))
