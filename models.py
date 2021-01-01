import keras.backend as K
from keras.layers import Dense, Flatten, Concatenate, Input, Dropout, Conv2D, MaxPooling2D, GlobalAveragePooling2D
from generators import DataGenerator
import numpy as np
import tensorflow as tf
import utils
from random_bins import bins_combiner_layer
from random_bins import binner

NUMBER_OF_RVC_CLASSES = 101
BINNER = binner.Binner(n_classes=NUMBER_OF_RVC_CLASSES)


def regression_predict(model, x, input_shape, batch_size=32, preprocessing_function=None, normalization_function=None):
    data_generator = DataGenerator(x, labels=None, input_shape=input_shape, batch_size=batch_size,
                                   preprocessing_function=preprocessing_function,
                                   normalization_function=normalization_function)

    y = model.predict(data_generator, verbose=1)  # predict should work as predict_generator if a generator is passed

    # y_pred = [[3, 7, 8], [5, 8, 9], [7, 6, 8]]

    # do not round to int
    return np.reshape(y, -1)


def rvc_predict(model, x, input_shape, batch_size=32, preprocessing_function=None, normalization_function=None):
    data_generator = DataGenerator(x, labels=None, input_shape=input_shape, batch_size=batch_size,
                                   preprocessing_function=preprocessing_function,
                                   normalization_function=normalization_function)

    y = model.predict(data_generator, verbose=1)  # predict should work as predict_generator if a generator is passed

    y_processed = tf.map_fn(lambda element: tf.math.argmax(element), y, dtype=tf.dtypes.int64)

    with tf.Session().as_default():
        return y_processed.eval()


def normalize_input_rcmalli(x, version, data_format=None):
    x_temp = x.astype(dtype="float32")
    if data_format is None:
        data_format = K.image_data_format()
    assert data_format in {'channels_last', 'channels_first'}

    # VGG16
    if version == 1:
        if data_format == 'channels_first':
            x_temp = x_temp[:, ::-1, ...]
            x_temp[:, 0, :, :] -= 93.5940
            x_temp[:, 1, :, :] -= 104.7624
            x_temp[:, 2, :, :] -= 129.1863
        else:
            x_temp = x_temp[..., ::-1]
            x_temp[..., 0] -= 93.5940
            x_temp[..., 1] -= 104.7624
            x_temp[..., 2] -= 129.1863

    # RESNET50, SENET50
    elif version == 2:
        if data_format == 'channels_first':
            x_temp = x_temp[:, ::-1, ...]
            x_temp[:, 0, :, :] -= 91.4953
            x_temp[:, 1, :, :] -= 103.8827
            x_temp[:, 2, :, :] -= 131.0912
        else:
            x_temp = x_temp[..., ::-1]
            x_temp[..., 0] -= 91.4953
            x_temp[..., 1] -= 103.8827
            x_temp[..., 2] -= 131.0912
    else:
        raise NotImplementedError

    return x_temp


def vgg16_normalization(dataset, **kwargs):
    return normalize_input_rcmalli(dataset, 1)


def resnet50_senet50_normalization(dataset, **kwargs):
    return normalize_input_rcmalli(dataset, 2)


def regression_output_function(last_layer):
    output = Dense(1, activation='relu', kernel_initializer='glorot_normal', name='regression')(
        last_layer)  # todo prob meglio HE_NORMAL
    loss = "mse"

    # Note, the metrics are not used to optimize the weights (the loss function is used)
    # They are used to judge the perfomance of the model, for example in the validation phase.

    # I have to consider to implement a custom metric that cast to int the input and then apply
    # the mae
    metrics = ['mae']

    return output, loss, metrics, "val_mae"


def rvc_output_function(last_layer):
    output = Dense(NUMBER_OF_RVC_CLASSES, activation='softmax', kernel_initializer='glorot_normal', name='rvc')(last_layer)

    loss = "categorical_crossentropy"

    metrics = [rvc_mae]

    return output, loss, metrics, "val_rvc_mae"


def rvc_mae(y_true, y_pred):
    y_true = tf.map_fn(lambda element: tf.math.argmax(element), y_true, dtype=tf.dtypes.int64)
    y_pred = tf.map_fn(lambda element: tf.math.argmax(element), y_pred, dtype=tf.dtypes.int64)

    return tf.keras.losses.MAE(y_true, y_pred)


def standard_dense_layer_structure(backbone):
    global_pool = GlobalAveragePooling2D()
    x = global_pool(backbone)

    # dense
    x = Dense(4096, activation='relu', kernel_initializer='he_normal')(x)
    x = Dense(2048, activation='relu', kernel_initializer='he_normal')(x)
    x = Dense(1024, activation='relu', kernel_initializer='he_normal')(x)

    return x


AVAILABLE_BACKENDS = ["vgg16", "resnet50", "senet50"]
AVAILABLE_FINAL_DENSE_STRUCTURE = {'standard_dense_layer_structure': standard_dense_layer_structure}
AVAILABLE_OUTPUT_TYPES = {"regression": regression_output_function,
                          'rvc': rvc_output_function,
                          'random_bins_classification': BINNER.architecture}

NORMALIZATION_FUNCTIONS = {"vgg16_normalization": vgg16_normalization,
                           "resnet50_normalization": resnet50_senet50_normalization,
                           "senet50_normalization": resnet50_senet50_normalization}
PREDICT_FUNCTIONS = {"regression_predict_function": regression_predict, "rvc_predict_function": rvc_predict}

CUSTOM_OBJECTS = {
    "rvc_mae": rvc_mae,
    'BinsCombinerLayer': bins_combiner_layer.BinsCombinerLayer
}
