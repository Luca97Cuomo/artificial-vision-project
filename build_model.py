# https://github.com/rcmalli/keras-vggface

# pip install git+https://github.com/rcmalli/keras-vggface.git

# You need tensorflow 1, does not work with tensorflow 2

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

def build_structure(backend, input):
    backend_out = backend(input)

    global_pool = GlobalAveragePooling2D()
    x = global_pool(backend_out)

    # dense
    x = Dense(4096, activation='relu', kernel_initializer='he_normal')(x)
    x = Dense(2048, activation='relu', kernel_initializer='he_normal')(x)
    x = Dense(1024, activation='relu', kernel_initializer='he_normal')(x)

    return x


def regression_output_function(last_layer):
    output = Dense(1, activation='relu', kernel_initializer='glorot_normal', name='regression')(last_layer)
    loss = "mse"

    # Note, the metrics are not used to optimize the weights (the loss function is used)
    # They are used to judge the perfomance of the model, for example in the validation phase.

    # I have to consider to implement a custom metric that cast to int the input and then apply
    # the mae
    metrics = [tf.keras.metrics.MeanAbsoluteError()]

    return output, loss, metrics, "mean_absolute_error"


AVAILABLE_BACKENDS = ["vgg16", "resnet50", "senet50"]
AVAILABLE_OUTPUT_TYPES = {"regression": regression_output_function}


def build_model(backend_name, output_type, output_dir, verbose=True):
    input_shape = 224, 224, 3

    backend = VGGFace(model=backend_name, include_top=False, input_shape=input_shape, weights='vggface')

    for layer in backend.layers:
        layer.trainable = False

    input = Input(shape=input_shape)

    optimizer = optimizers.Adam(lr=0.0005) # lr is an hyperparameter
    output_function = AVAILABLE_OUTPUT_TYPES[output_type]

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
        metadata = {"val_metric_name": val_metric_name}
        f.write(json.dumps(metadata)) # use `json.loads` to do the reversese

    if verbose:
        model.summary()
        plot_model(model)


def main():
    parser = argparse.ArgumentParser(description='Build model')
    parser.add_argument('-b', '--backend_name', type=str, help='The name of the backend to use (vgg16, resnet50, senet50)', required=True)
    parser.add_argument('-o', '--output_type', type=str, help='The output type of the network (regression, RvC, multiRvC)', required=True)
    parser.add_argument('-m', '--model_path', type=str, help='The path where to save the compiled model', required=True)
    parser.add_argument('-v', '--verbose',  action='store_true', help='Verbose')

    args = parser.parse_args()

    build_model(args.backend_name, args.output_type, args.model_path, args.verbose)

if __name__ == '__main__':
    main()



