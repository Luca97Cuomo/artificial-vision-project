# https://github.com/rcmalli/keras-vggface

# pip install git+https://github.com/rcmalli/keras-vggface.git

# You need tensorflow 1, does not work with tensorflow 2

import argparse
import os
from keras import optimizers
from keras.models import Model
from keras.layers import Dense, Flatten, Concatenate, Input, Dropout, Conv2D, MaxPooling2D, GlobalAveragePooling2D
from pathlib import Path
import tensorflow as tf
if tf.__version__[0] != "2":
    from keras_vggface.vggface import VGGFace
import models
import configuration


def build_structure(input, backend, dense_layer_structure):
    backend_out = backend(input)

    x = dense_layer_structure(backend_out)

    return x


def build_model(configuration_file_path):
    conf = configuration.read_configuration(configuration_file_path)

    build = conf["build"]

    backend = build["backend"]
    backend_name = backend["name"]
    unlock_layers = backend["unlock_layers"]

    input_shape = conf["input_shape"]
    output_type = conf["output_type"]
    output_dir = build["build_model_dir"]
    learning_rate = build["build_learning_rate"]
    dense_layer_structure_name = build["dense_layer_structure_name"]
    model_name = conf["model_name"]

    verbose = conf["verbose"]

    if backend_name not in models.AVAILABLE_BACKENDS:
        raise Exception("The backend is not available")

    normalization_function_name = backend_name + "_normalization"
    if normalization_function_name not in models.NORMALIZATION_FUNCTIONS:
        raise Exception("The normalization function is not available")

    predict_function_name = output_type + "_predict_function"
    if predict_function_name not in models.PREDICT_FUNCTIONS:
        raise Exception("The predict function is not available")

    if dense_layer_structure_name not in models.AVAILABLE_FINAL_DENSE_STRUCTURE:
        raise Exception("The dense layer structure is not available")

    if backend_name == "vgg19":
        backend = tf.keras.applications.VGG19(include_top=False, weights="imagenet")
    else:
        backend = VGGFace(model=backend_name, include_top=False, input_shape=input_shape, weights='vggface')

    if unlock_layers == "none":
        for layer in backend.layers:
            layer.trainable = False
    elif unlock_layers == "all":
        for layer in backend.layers:
            layer.trainable = True
    else:
        for layer in backend.layers[:-unlock_layers]:
            layer.trainable = False

    model_input = Input(shape=input_shape)

    optimizer = optimizers.Adam(lr=learning_rate)  # lr is an hyperparameter

    dense_layer_structure = models.AVAILABLE_FINAL_DENSE_STRUCTURE[dense_layer_structure_name]
    output_function = models.AVAILABLE_OUTPUT_TYPES[output_type]

    last_layer = build_structure(model_input, backend, dense_layer_structure)

    output, loss, metrics, val_metric_name = output_function(last_layer)

    model = Model(inputs=model_input, outputs=output, name=model_name)
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    model_dir = os.path.join(output_dir, model_name)
    Path(model_dir).mkdir(parents=True, exist_ok=True)
    model.save(os.path.join(model_dir, model_name + "_model"))

    # Saving metadata
    metadata = {"monitored_quantity": val_metric_name,
                "normalization_function_name": normalization_function_name,
                "predict_function_name": predict_function_name,
                "input_shape": input_shape,
                "output_type": output_type}

    configuration.save_configuration(conf["metadata_path"], metadata)

    if verbose:
        model.summary()
        print("Backend structure:\n")
        for layer in backend.layers:
            print(f"{layer.name}: trainable {layer.trainable}")


def main():
    parser = argparse.ArgumentParser(description='Build model')
    parser.add_argument('-c', '--configuration_file_path', type=str, help='The path of the configuration file',
                        required=True)

    args = parser.parse_args()

    build_model(args.configuration_file_path)


if __name__ == '__main__':
    main()
