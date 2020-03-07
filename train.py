import argparse
import os
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import save_model
from tensorflow.python.keras import Input
from tensorflow.python.keras.models import Model

from config import config
from label_codec import LabelCodec
from train_helper import TrainHelper
from licence_plate_dataset_generator import LicensePlateDatasetGenerator
from license_plate_image_augmentor import LicensePlateImageAugmentor
from utils.io.hdf5datasetloader import Hdf5DatasetLoader
from utils.nn.conv import OCR

ap = argparse.ArgumentParser()
ap.add_argument("-o", "--optimizer", default=config.OPTIMIZER, help="supported optimizer methods: sdg, rmsprop, adam, adagrad, adadelta")
args = vars(ap.parse_args())

OPTIMIZER = args["optimizer"]
MODEL_PATH = os.path.join(config.OUTPUT_PATH, OPTIMIZER, config.MODEL_NAME) + ".h5"
MODEL_WEIGHTS_PATH = os.path.join(config.OUTPUT_PATH, OPTIMIZER, config.MODEL_NAME) + '-weights.h5'

os.makedirs(os.path.join(config.OUTPUT_PATH, OPTIMIZER), exist_ok=True)

print("Optimizer:    {}".format(OPTIMIZER))
print("Weights path: {}".format(MODEL_WEIGHTS_PATH))
print("Model path:   {}".format(MODEL_PATH))

tf.compat.v1.disable_eager_execution()
# tf.compat.v1.disable_v2_behavior()


print("[INFO] loading data...")
loader = Hdf5DatasetLoader()
images, labels = loader.load(config.GLP_HDF5, shuffle=True)
X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2)

background_images = loader.load(config.BACKGRND_HDF5, shuffle=True, max_items=10000)

augmentator = LicensePlateImageAugmentor(config.IMAGE_WIDTH, config.IMAGE_HEIGHT, background_images)

train_generator = LicensePlateDatasetGenerator(X_train, y_train, config.IMAGE_WIDTH, config.IMAGE_HEIGHT,
                                               config.DOWNSAMPLE_FACTOR, config.MAX_TEXT_LEN, config.BATCH_SIZE,
                                               augmentator)

val_generator = LicensePlateDatasetGenerator(X_val, y_val, config.IMAGE_WIDTH, config.IMAGE_HEIGHT,
                                             config.DOWNSAMPLE_FACTOR, config.MAX_TEXT_LEN, config.BATCH_SIZE,
                                             augmentator)

test_generator = LicensePlateDatasetGenerator(X_test, y_test, config.IMAGE_WIDTH, config.IMAGE_HEIGHT,
                                              config.DOWNSAMPLE_FACTOR, config.MAX_TEXT_LEN, config.BATCH_SIZE,
                                              augmentator)

print("Train dataset size:      {}".format(X_train.shape[0]))
print("Validation dataset size: {}".format(X_val.shape[0]))
print("Test dataset size:       {}".format(X_test.shape[0]))


class CTCLoss(tf.keras.losses.Loss):

    def __init__(self, input_length, label_length, name='CTCLoss'):
        super().__init__(name=name)
        self.input_length = input_length
        self.label_length = label_length

    def call(self, labels, predictions):
        return tf.keras.backend.ctc_batch_cost(labels, predictions, self.input_length, self.label_length)


print("[INFO] build model...")
labels = Input(name='labels', shape=(config.MAX_TEXT_LEN,), dtype='float32')
input_length = Input(name='input_length', shape=(1,), dtype='int64')
label_length = Input(name='label_length', shape=(1,), dtype='int64')

inputs, outputs = OCR.conv_bgru((config.IMAGE_WIDTH, config.IMAGE_HEIGHT, 1), len(LabelCodec.ALPHABET) + 1)
train_model = Model(inputs=[inputs, labels, input_length, label_length], outputs=outputs)
train_model.add_loss(CTCLoss(input_length, label_length)(labels, outputs))
train_model.compile(loss=None, optimizer=TrainHelper.get_optimizer(OPTIMIZER))

print("[INFO] model architecture...")
train_model.summary()

print("[INFO] train model...")
history = train_model.fit(
    train_generator.generator(),
    steps_per_epoch=train_generator.numImages // config.BATCH_SIZE,
    validation_data=val_generator.generator(),
    validation_steps=val_generator.numImages // config.BATCH_SIZE,
    epochs=config.NUM_EPOCHS,
    callbacks=TrainHelper.get_callbacks(OPTIMIZER, MODEL_WEIGHTS_PATH), verbose=1)

print("[INFO] save model...")
predict_model = Model(inputs=inputs, outputs=outputs)
predict_model.load_weights(MODEL_WEIGHTS_PATH)
save_model(predict_model, filepath=MODEL_PATH, save_format="h5")

print("[INFO] plot and save training history...")
print(history.history.keys())
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.savefig(os.path.join(config.DOCUMENTATION_PATH, config.MODEL_NAME) + "-train-history-" + OPTIMIZER + ".png")
plt.show()

print("[INFO] evaluating model...")
X_test, y_test = next(test_generator.generator())
score = predict_model.evaluate(X_test, y_test, verbose=0)
print('Test loss: {0:.4f}'.format(score[0]))
