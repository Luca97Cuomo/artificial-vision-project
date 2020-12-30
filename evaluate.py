from utils import *
from preprocessing import load_labels
import json
import keras
import numpy as np
import preprocessing_functions
import argparse

import configuration
from models import NORMALIZATION_FUNCTIONS
from models import PREDICT_FUNCTIONS


def evaluate(Y, Y_pred):
    """
    Y and Y_pred have to be list or numpy arrays
    """

    mae_int = np.average(abs(np.rint(Y) - np.rint(Y_pred)))
    mae_float = np.mean(abs(Y - Y_pred))
    return mae_int, mae_float


def evaluate_model(configuration_file_path):
# def evaluate_model(model_path, metadata_path, preprocessing_function, x_test, y_test, batch_size, output_path=None):
    conf = configuration.read_configuration(configuration_file_path)

    model_path = conf["train_model_path"]
    preprocessing_function_name = conf["preprocessing_function_name"]
    enable_preprocessing = conf["enable_preprocessing"]
    test_set_path = conf["test_set_path"]
    num_test_samples = conf["num_test_samples"]
    batch_size = conf["batch_size"]
    save_predictions = conf["save_prediction"]
    save_predictions_path = conf["save_predictions_path"]
    csv_path = conf["csv_path"]
    predict_function_name = conf["predict_function_name"]
    input_shape = conf["input_shape"]
    normalization_function_name = conf["normalization_function_name"]

    if not enable_preprocessing:
        preprocessing_function = None
    elif preprocessing_function_name not in preprocessing_functions.AVAILABLE_PREPROCESSING_FUNCTIONS:
        raise Exception("The requested preprocessing function is not supported")
    else:
        preprocessing_function = preprocessing_functions.AVAILABLE_PREPROCESSING_FUNCTIONS[preprocessing_function_name]

    labels_dict = load_labels(csv_path, False)
    x_test, y_test = prepare_data_for_generator(test_set_path, labels_dict, num_test_samples)

    model = keras.models.load_model(model_path)

    predict_function = PREDICT_FUNCTIONS[predict_function_name]
    normalization_function = NORMALIZATION_FUNCTIONS[normalization_function_name]

    y_pred = predict_function(model, x_test, input_shape=input_shape, batch_size=batch_size,
                              preprocessing_function=preprocessing_function,
                              normalization_function=normalization_function)

    mae_int, mae_float = evaluate(y_test, y_pred)
    print(f"MAE int: {mae_int}\nMAE float {mae_float}")

    # saving predictions if the path is not None
    if save_predictions:
        print(f"Saving predictions to {save_predictions_path}")
        with open(save_predictions_path, 'w') as f:
            for i in range(len(x_test)):
                x_splitted = x_test[i].split("/")
                identity = x_splitted[-2]
                image = x_splitted[-1]
                path = identity + "/" + image
                f.write(f'{path},{int(round(y_pred[i]))},{int(round(y_test[i]))}\r\n')
        print("Predictions saved")


def main():
    parser = argparse.ArgumentParser(description='Evaluate model')
    parser.add_argument('-c', '--configuration_file_path', type=str, help='The path of the configuration file', required=True)

    args = parser.parse_args()

    evaluate_model(args.configuration_file_path)


if __name__ == '__main__':
    main()
