# https://github.com/rcmalli/keras-vggface

# pip install git+https://github.com/rcmalli/keras-vggface.git

# You need tensorflow 1, does not work with tensorflow 2

import argparse
import os
import tensorflow as tf
from keras import optimizers
from keras.models import Model
from keras.layers import Dense, Flatten, Concatenate, Input, Dropout, Conv2D, MaxPooling2D, GlobalAveragePooling2D
from keras_vggface.vggface import VGGFace
import json
from pathlib import Path

import models
import configuration

# width, height, channels
INPUT_SHAPE = 224, 224, 3


def build_structure(backend, input):
    backend_out = backend(input)

    global_pool = GlobalAveragePooling2D()
    x = global_pool(backend_out)

    # dense
    x = Dense(4096, activation='relu', kernel_initializer='he_normal')(x)
    x = Dense(2048, activation='relu', kernel_initializer='he_normal')(x)
    x = Dense(1024, activation='relu', kernel_initializer='he_normal')(x)

    return x


def build_model(configuration_file_path):
    conf = configuration.read_configuration(configuration_file_path)

    backend_name = conf["backend_name"]
    output_type = conf["output_type"]
    output_dir = conf["build_model_dir"]
    learning_rate = conf["build_learning_rate"]
    verbose = conf["verbose"]

    normalization_function_name = backend_name + "_normalization"
    if normalization_function_name not in models.NORMALIZATION_FUNCTIONS:
        raise Exception("The normalization function is not available")

    predict_function_name = output_type + "_predict_function"
    if predict_function_name not in models.PREDICT_FUNCTIONS:
        raise Exception("The predict function is not available")

    backend = VGGFace(model=backend_name, include_top=False, input_shape=INPUT_SHAPE, weights='vggface')

    for layer in backend.layers:
        layer.trainable = False

    model_input = Input(shape=INPUT_SHAPE)

    optimizer = optimizers.Adam(lr=learning_rate)  # lr is an hyperparameter
    output_function = models.AVAILABLE_OUTPUT_TYPES[output_type]

    last_layer = build_structure(backend, model_input)
    output, loss, metrics, val_metric_name = output_function(last_layer)

    model_name = backend_name + "_" + output_type

    model = Model(inputs=model_input, outputs=output, name=model_name)
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    model_dir = os.path.join(output_dir, model_name)
    Path(model_dir).mkdir(parents=True, exist_ok=True)
    model.save(os.path.join(model_dir, model_name + "_model"))

    conf["val_metric_name"] = val_metric_name
    conf["normalization_function_name"] = normalization_function_name
    conf["predict_function_name"] = predict_function_name
    conf["input_shape"] = INPUT_SHAPE
    with open(configuration_file_path, "w") as f:
        f.write(json.dumps(conf))

    if verbose:
        model.summary()


def main():
    parser = argparse.ArgumentParser(description='Build model')
    parser.add_argument('-c', '--configuration_file_path', type=str, help='The path of the configuration file', required=True)

    args = parser.parse_args()

    build_model(args.configuration_file_path)


if __name__ == '__main__':
    main()
