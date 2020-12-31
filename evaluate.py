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
from data_analysis import get_age_interval


def evaluate_by_age_intervals(age_interval_width, y, y_pred, verbose=True):
    for i in range(len(y)):

    ages = {}
    count = 0

    max_age = -1
    min_age = 1000



    for identity in labels:
        for image in labels[identity]:
            age = labels[identity][image]

            if age < min_age:
                min_age = age
            if age > max_age:
                max_age = age

            interval_start_age = get_age_interval(age, age_interval_width, 0)

            if interval_start_age in ages:
                ages[interval_start_age] += 1
            else:
                ages[interval_start_age] = 1
            count += 1
            if verbose:
                if count % 10000 == 0:
                    print("Processed " + str(count) + " labels")

    ordered_dict = collections.OrderedDict(sorted(ages.items()))
    for interval_start_age in ordered_dict:
        print("age interval: [" + str(interval_start_age) + ",  " + str(interval_start_age + age_interval_width - 1) + "] - occurrences: " + str(ordered_dict[interval_start_age]) +" - percentage: " + str(ordered_dict[interval_start_age] / count))



def evaluate(Y, Y_pred, verbose=True):
    """
    Y and Y_pred have to be list or numpy arrays
    """

    mae_int = np.average(abs(np.rint(Y) - np.rint(Y_pred)))
    mae_float = np.mean(abs(Y - Y_pred))

    print(f"MAE int: {mae_int}")
    print(f"MAE float: {mae_float}")

    return mae_int, mae_float


def evaluate_model(configuration_file_path):
# def evaluate_model(model_path, metadata_path, preprocessing_function, x_test, y_test, batch_size, output_path=None):
    conf = configuration.read_configuration(configuration_file_path)

    model_path = conf["model_path"]

    preprocessing = conf["preprocessing"]
    evaluate = ["evaluate"]

    preprocessing_function_name = preprocessing["preprocessing_function_name"]
    enable_preprocessing = preprocessing["enable"]

    test_set_path = evaluate["test_set_path"]
    num_test_samples = evaluate["num_test_samples"]

    batch_size = conf["batch_size"]

    save_predictions_dict = evaluate["save_predictions"]

    save_predictions = save_predictions_dict["save_prediction"]
    save_predictions_path = save_predictions_dict["save_predictions_path"]

    predict_function_name = evaluate["predict_function_name"]

    normalization_function_name = conf["normalization_function_name"]

    csv_path = conf["csv_path"]
    input_shape = conf["input_shape"]

    age_intervals_evaluation = evaluate["age_intervals_evaluation"]

    evaluate_by_age_intervals = age_intervals_evaluation["enabled"]
    age_interval_width = age_intervals_evaluation["age_interval_width"]

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

    evaluate(y_test, y_pred, True)

    if evaluate_by_age_intervals:
        evaluate_by_age_intervals(age_interval_width, y_test, y_pred, True)

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
