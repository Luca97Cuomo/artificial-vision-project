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

    # y = [[3], [5], [8]]

    # do not round to int
    return np.reshape(y, -1)


def rvc_predict(model, x, input_shape, batch_size=32, preprocessing_function=None, normalization_function=None):
    data_generator = DataGenerator(x, labels=None, input_shape=input_shape, batch_size=batch_size,
                                   preprocessing_function=preprocessing_function,
                                   normalization_function=normalization_function)

    y = model.predict(data_generator, verbose=1)  # predict should work as predict_generator if a generator is passed

    # y = [[3, 6, 8], [5, 7, 0], [8, 4, 7]]

    y_processed = tf.map_fn(lambda element: tf.math.argmax(element), y, dtype=tf.dtypes.int64)

    with tf.Session().as_default():
        return y_processed.eval()


def random_bins_classification_predict(model, x, input_shape, batch_size=32, preprocessing_function=None, normalization_function=None):
    data_generator = DataGenerator(x, labels=None, input_shape=input_shape, batch_size=batch_size,
                                   preprocessing_function=preprocessing_function,
                                   normalization_function=normalization_function)

    y = model.predict(data_generator, verbose=1)  # predict should work as predict_generator if a generator is passed

    if len(x) != len(y[0]):
        raise Exception("Unexpected number of predictions")

    return from_random_bins_classification_to_regression(y, BINNER)


def from_random_bins_classification_to_regression(y, binner):
    """
    :param y:       list or numpy array
    :param binner:
    :return:        numpy array
    """
    means = binner.compute_means()

    y_pred = []

    # the shape of y is [n_classifiers, n_samples, n_intervals]
    # so the n_samples is the second parameter, not the first.

    if len(means) != len(y):
        raise Exception("Unexpected number of classifiers")

    # for each sample
    for i in range(len(y[0])):
        sum = 0

        # len(means) represents the number of classifiers
        for j in range(len(means)):
            output = y[j][i]
            mean = means[j]

            index = np.argmax(output)
            max_prob = output[index]

            # age = mean[index] * max_prob
            age = mean[index]

            sum += age

        mean_age = sum / len(means)
        y_pred.append(mean_age)

    return np.array(y_pred)


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


def vgg19_normalization(dataset):
    dataset = dataset.astype(dtype="float32")
    return tf.keras.applications.vgg19.preprocess_input(dataset)


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
    output = Dense(NUMBER_OF_RVC_CLASSES, activation='softmax', kernel_initializer='glorot_normal', name='rvc')(
        last_layer)

    loss = "categorical_crossentropy"

    metrics = [rvc_mae]

    return output, loss, metrics, "val_rvc_mae"


def rvc_mae(y_true, y_pred):
    y_true = tf.map_fn(lambda element: tf.math.argmax(element), y_true, dtype=tf.dtypes.int64)
    y_pred = tf.map_fn(lambda element: tf.math.argmax(element), y_pred, dtype=tf.dtypes.int64)

    return tf.keras.losses.MAE(tf.dtypes.cast(y_true, dtype=tf.dtypes.float64),
                               tf.dtypes.cast(y_pred, dtype=tf.dtypes.float64))


# def bins_classification_mae(y_true, y_pred):
#     # [classes, samples]
#
#     y_pred_regression = from_random_bins_classification_to_regression(from_tensor_to_numpy(y_pred), BINNER)
#
#     return tf.keras.losses.MAE(tf.dtypes.cast(y_true, dtype=tf.dtypes.float64),
#                                tf.convert_to_tensor(y_pred_regression, dtype=tf.dtypes.float64))
#
#
# def bins_classification_loss(y_true, y_pred):
#     # [classes, samples]
#     binned_labels = BINNER.bin_labels(from_tensor_to_numpy(y_true))
#
#     if BINNER.n_interval_sets != len(binned_labels):
#         raise Exception("n_interval_sets != len(binned_labels)")
#
#     cce_function = tf.keras.losses.CategoricalCrossentropy()
#
#     sum = cce_function(binned_labels[0], from_tensor_to_numpy(y_pred[0]))
#     for i in range(1, BINNER.n_interval_sets):
#         sum += cce_function(binned_labels[i], from_tensor_to_numpy(y_pred[i]))
#
#     return sum


def standard_dense_layer_structure(backbone):
    global_pool = GlobalAveragePooling2D()
    x = global_pool(backbone)

    # dense
    x = Dense(4096, activation='relu', kernel_initializer='he_normal')(x)
    x = Dense(2048, activation='relu', kernel_initializer='he_normal')(x)
    x = Dense(1024, activation='relu', kernel_initializer='he_normal')(x)

    return x


def vgg16_dense_layer_structure(backbone):
    global_pool = GlobalAveragePooling2D()
    x = global_pool(backbone)

    # dense
    x = Dense(4096, activation='relu', kernel_initializer='he_normal')(x)
    x = Dense(4096, activation='relu', kernel_initializer='he_normal')(x)

    return x

#
# def from_tensor_to_numpy(y):
#     with tf.Session().as_default():
#         return y.eval()


AVAILABLE_BACKENDS = ["vgg16", "resnet50", "senet50", "vgg19"]


AVAILABLE_FINAL_DENSE_STRUCTURE = {'standard_dense_layer_structure': standard_dense_layer_structure,
                                   'vgg16_dense_layer_structure': vgg16_dense_layer_structure}


AVAILABLE_OUTPUT_TYPES = {"regression": regression_output_function,
                          'rvc': rvc_output_function,
                          'random_bins_classification': BINNER.architecture}

NORMALIZATION_FUNCTIONS = {"vgg16_normalization": vgg16_normalization,
                           "resnet50_normalization": resnet50_senet50_normalization,
                           "senet50_normalization": resnet50_senet50_normalization,
                           "vgg19_normalization": vgg19_normalization
                           }

PREDICT_FUNCTIONS = {"regression_predict_function": regression_predict,
                     "rvc_predict_function": rvc_predict,
                     "random_bins_classification_predict_function": random_bins_classification_predict}


CUSTOM_OBJECTS = {
    "rvc_mae": rvc_mae,
    'BinsCombinerLayer': bins_combiner_layer.BinsCombinerLayer,
    # "bins_classification_loss": bins_classification_loss,
    # "bins_classification_mae": bins_classification_mae
}
