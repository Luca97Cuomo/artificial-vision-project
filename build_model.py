# https://github.com/rcmalli/keras-vggface

# pip install git+https://github.com/rcmalli/keras-vggface.git

# You need tensorflow 1, does not work with tensorflow 2

# %tensorflow_version 1.x

import argparse
import os
import tensorflow as tf
from keras.utils import plot_model
from keras import optimizers
from keras.models import Model
from keras.layers import Dense, Flatten, Concatenate, Input, Dropout, Conv2D, MaxPooling2D, GlobalAveragePooling2D
from keras_vggface.vggface import VGGFace
import json
from pathlib import Path
import models

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


def build_model(backend_name, output_type, output_dir, verbose=True):
    normalization_function_name = backend_name + "_normalization"
    if normalization_function_name not in models.NORMALIZATION_FUNCTIONS:
        raise Exception("The normalization function is not available")

    predict_function_name = output_type + "_predict_function"
    if predict_function_name not in models.PREDICT_FUNCTIONS:
        raise Exception("The predict function is not available")

    backend = VGGFace(model=backend_name, include_top=False, input_shape=INPUT_SHAPE, weights='vggface')

    for layer in backend.layers:
       layer.trainable = False

    input = Input(shape=INPUT_SHAPE)

    optimizer = optimizers.Adam(lr=0.0005)  # lr is an hyperparameter
    output_function = models.AVAILABLE_OUTPUT_TYPES[output_type]

    last_layer = build_structure(backend, input)
    output, loss, metrics, val_metric_name = output_function(last_layer)

    model_name = backend_name + "_" + output_type

    model = Model(inputs=input, outputs=output, name=model_name)
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    model_dir = os.path.join(output_dir, model_name)
    Path(model_dir).mkdir(parents=True, exist_ok=True)
    model.save(os.path.join(model_dir, model_name + "_model"))
    meta_file_name = os.path.join(model_dir, model_name + "_metadata.txt")
    with open(meta_file_name, "w") as f:
        metadata = {"val_metric_name": val_metric_name,
                    "normalization_function_name": normalization_function_name,
                    "predict_function_name": predict_function_name,
                    "input_shape": INPUT_SHAPE}
        f.write(json.dumps(metadata))  # use `json.loads` to do the reversese

    if verbose:
        model.summary()
        # plot_model(model)


def main():
    parser = argparse.ArgumentParser(description='Build model')
    parser.add_argument('-b', '--backend_name', type=str,
                        help='The name of the backend to use (vgg16, resnet50, senet50)', required=True)
    parser.add_argument('-o', '--output_type', type=str,
                        help='The output type of the network (regression, RvC, multiRvC)', required=True)
    parser.add_argument('-m', '--model_path', type=str, help='The path where to save the compiled model', required=True)
    parser.add_argument('-v', '--verbose', action='store_true', help='Verbose')

    args = parser.parse_args()

    if args.backend_name not in models.AVAILABLE_BACKENDS:
        raise Exception("The requested backend is not supported")

    if args.output_type not in models.AVAILABLE_OUTPUT_TYPES:
        raise Exception("The requested output type is not supported")

    print(tf.version)
    build_model(args.backend_name, args.output_type, args.model_path, args.verbose)


if __name__ == '__main__':
    main()
