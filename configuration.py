import argparse
import json


def save_configuration_template(path, verbose=False):
    configuration = {
        "output_type": "regression",

        "model_name": "vgg16_regression_model",
        "model_path": "/models/vgg16_regression/vgg16_regression_model",
        "csv_path": "/train.age_detected.csv",

        "batch_size": 256,

        "normalization_function_name": "vgg16_normalization",

        "input_shape": (224, 224, 3),

        "verbose": True,

        "preprocessing": {
            "enable": False,
            "preprocessing_function_name": "standard_preprocessing_function",
        },

        "build": {
            "build_model_dir": "/models/",
            "build_learning_rate": 0.0001,

            "dense_layer_structure_name": "standard_dense_layer_structure",

            "backend": {
                "name": "vgg16",
                "unlock_layers": "none" # ["none", "all", 1, 2, ...]
            },
        },

        "train": {
            "training_set_path": "/content/training_set_resized",
            "validation_set_path": "/content/validation_set_resized",
            "num_training_samples": 300000,
            "num_validation_samples": 30000,
            "monitored_quantity": "val_mae",
            "augmentations": False,
            "epochs": 20,
            "initial_epoch": 0,
            # train_learning_rate: if not None the learning rate of model will be changed
            "train_learning_rate": None,
            "output_training_dir": "/models/vgg16_regression/"
        },

        "evaluate": {
            "test_set_path": "/content/test_set_resized",
            "num_test_samples": 100000,

            "save_predictions": {
                "enabled": False,
                "save_predictions_path": "/models/vgg16_regression/predictions.txt"
            },

            "age_intervals_evaluation": {
                "enabled": True,
                "age_interval_width": 10
            },

            "predict_function_name": "regression_predict"

        },
    }

    with open(path, 'w') as f:
        f.write(json.dumps(configuration, indent=2))

    if verbose:
        dump_configuration(configuration)


def read_configuration(path):
    with open(path, 'r') as f:
        configuration = json.load(f)

    if configuration["verbose"]:
        dump_configuration(configuration)

    return configuration


def save_configuration(path, configuration):
    with open(path, 'w') as f:
        f.write(json.dumps(configuration, indent=2))


def dump_configuration(configuration):
    print(json.dumps(configuration, indent=2))


def main():
    parser = argparse.ArgumentParser(description='Configuration')
    parser.add_argument('-p', '--path', type=str, help='Tha path of the file where to save the configuration template',
                        required=True)
    parser.add_argument('-v', '--verbose', action='store_true', help='Verbose')

    args = parser.parse_args()

    save_configuration_template(args.path, args.verbose)


if __name__ == '__main__':
    main()
