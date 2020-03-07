import os

from tensorflow.keras.optimizers import SGD, Adam, Adagrad, Adadelta, RMSprop
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.python.keras.callbacks import TensorBoard, ModelCheckpoint

from config import config


class TrainHelper:
    @staticmethod
    def get_optimizer(optimizer):
        if optimizer == "sdg":
            return SGD(learning_rate=0.01, decay=1e-6, momentum=0.9, nesterov=True, clipnorm=5)
        if optimizer == "rmsprop":
            return RMSprop(learning_rate=0.01)
        if optimizer == "adam":
            return Adam(learning_rate=0.01)
        if optimizer == "adagrad":
            return Adagrad(learning_rate=0.01)
        if optimizer == "adadelta":
            return Adadelta(learning_rate=1.0)

    @staticmethod
    def get_callbacks(optimizer, model_weigths_path):
        logdir = os.path.join("logs", optimizer)
        chkpt_filepath = config.MODEL_NAME + '--{epoch:02d}--{loss:.3f}--{val_loss:.3f}.h5'

        callbacks = [
            EarlyStopping(monitor='val_loss', min_delta=0.0001, patience=4, verbose=1),
            ModelCheckpoint(filepath=model_weigths_path, monitor='val_loss', save_best_only=True, save_weights_only=True, verbose=1),
            TensorBoard(log_dir=logdir)]

        if optimizer in ["sdg", "rmsprop"]:
            callbacks.append(ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=2, verbose=1, mode='min', min_delta=0.01, cooldown=0, min_lr=0))

        return callbacks
