import numpy as np
import keras.backend as K
from keras.layers import Dense, Flatten, Concatenate, Input, Dropout, Conv2D, MaxPooling2D, GlobalAveragePooling2D
from generators import PredictDataGenerator


def regression_predict_function(model, X, preprocessing_function=None):
    data_generator = PredictDataGenerator(X, preprocessing_function=preprocessing_function)

    Y = model.predict_generator(data_generator)

    # do not round to int
    return Y


def normalize_input_rcmalli(x, version, data_format=None):
    x_temp = np.copy(x, dtype="float32")
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
    return normalize_input_rcmalli(dataset, 1, kwargs["data_format"])


def resnet50_senet50_normalization(dataset, **kwargs):
    return normalize_input_rcmalli(dataset, 2, kwargs["data_format"])


def regression_output_function(last_layer):
    output = Dense(1, activation='relu', kernel_initializer='glorot_normal', name='regression')(last_layer)
    loss = "mse"

    # Note, the metrics are not used to optimize the weights (the loss function is used)
    # They are used to judge the perfomance of the model, for example in the validation phase.

    # I have to consider to implement a custom metric that cast to int the input and then apply
    # the mae
    metrics = ['mae']

    return output, loss, metrics, "val_mae"


AVAILABLE_BACKENDS = ["vgg16", "resnet50", "senet50"]
AVAILABLE_OUTPUT_TYPES = {"regression": regression_output_function}
NORMALIZATION_FUNCTIONS = {"vgg16_normalization": vgg16_normalization, "resnet50_normalization": resnet50_senet50_normalization,
                           "senet50_normalization": resnet50_senet50_normalization}
PREDICT_FUNCTIONS = {"regression_predict_function": regression_predict_function}