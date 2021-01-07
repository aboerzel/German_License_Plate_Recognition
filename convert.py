import tensorflow as tf

MODEL_PATH = '../output/license_recognition/adagrad/glpr-model.h5'
TFLITE_MODEL_PATH = '../output/license_recognition/adagrad/glpr-model-float.tflite'
keras_model = tf.keras.models.load_model(MODEL_PATH)
converter = tf.lite.TFLiteConverter.from_keras_model(keras_model)
converter.experimental_new_converter = True
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_types = [tf.float16]
tflite_fp16_model = converter.convert()

open(TFLITE_MODEL_PATH, "wb").write(tflite_fp16_model)
