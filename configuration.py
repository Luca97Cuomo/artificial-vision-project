import argparse
import json


def save_configuration_template(path, verbose=False):
    configuration = {"backend_name": "vgg16",
                     "output_type": "regression",

                     "build_model_dir": "/models/",
                     "model_path": "/models/vgg16_regression/vgg16_regression_model",
                     "output_training_dir": "/models/vgg16_regression/",
                     "csv_path": "/train.age_detected.csv",

                     "build_learning_rate": 0.0001,
                     # train_learning_rate: if not None the learning rate of model will be changed
                     "train_learning_rate": None,
                     "batch_size": 256,
                     "epochs": 20,
                     "initial_epoch": 0,
                     "augmentations": False,

                     # old metadata parameters
                     # They are set by the build_model script
                     "val_metric_name": "val_mae",
                     "normalization_function_name": "vgg16_normalization",
                     "predict_function_name": "regression_predict",
                     "input_shape": (224, 224, 3),

                     "training_set_path": "/content/training_set_resized",
                     "validation_set_path": "/content/validation_set_resized",
                     "test_set_path": "/content/test_set_resized",

                     "num_training_samples": 300000,
                     "num_validation_samples": 30000,
                     "num_test_samples": 100000,

                     "enable_preprocessing": False,
                     "preprocessing_function_name": "standard_preprocessing_function",

                     "save_predictions": False,
                     "save_predictions_path": "/models/vgg16_regression/predictions.txt",

                     "verbose": True}

    with open(path, 'w') as f:
        f.write(json.dumps(configuration, indent=2))

    if verbose:
        print(configuration)


def read_configuration(path):
    with open(path, 'r') as f:
        configuration = json.load(f)

    if configuration["verbose"]:
        print(configuration)

    return configuration


def save_configuration(path, configuration):
    with open(path, 'w') as f:
        f.write(json.dumps(configuration, indent=2))


def main():
    parser = argparse.ArgumentParser(description='Configuration')
    parser.add_argument('-p', '--path', type=str, help='Tha path of the file where to save the configuration template', required=True)
    parser.add_argument('-v', '--verbose', action='store_true', help='Verbose')

    args = parser.parse_args()

    save_configuration_template(args.path, args.verbose)


if __name__ == '__main__':
    main()




