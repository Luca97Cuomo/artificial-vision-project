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
from models import CUSTOM_OBJECTS
from data_analysis import get_age_interval


def take_error_list_key(element):
    return element["start_age"]


def print_error_by_age_intervals(error_list):
    error_list.sort(key=take_error_list_key)

    print("")
    for i in range(len(error_list)):
        element = error_list[i]

        age_interval_label = element["age_interval_label"]
        mae =  element["mae"]
        overestimate_mae = element["overestimate_mae"]
        underestimate_mae = element["underestimate_mae"]
        count = element["count"]
        overestimate_count = element["overestimate_count"]
        underestimate_count = element["underestimate_count"]

        print(f"interval: {age_interval_label}")
        print(f"mae: {mae}")
        print(f"overestimate_mae: {overestimate_mae}")
        print(f"underestimate_mae: {underestimate_mae}")
        print(f"count: {count}")
        print(f"overestimate_count: {overestimate_count}")
        print(f"underestimate_count: {underestimate_count}")
        print("")


def evaluate_by_age_intervals(age_interval_width, y, y_pred, verbose=True):
    error_by_interval = {}
    for i in range(len(y)):
        interval_start_age = get_age_interval(int(round(y[i])), age_interval_width, 0)

        error = round(y[i]) - round(y_pred[i])
        abs_error = abs(error)

        current = error_by_interval.setdefault(interval_start_age, {"error_sum": 0,
                                                                    "count": 0,
                                                                    "overestimate_error_sum": 0,
                                                                    "underestimate_error_sum": 0,
                                                                    "overestimate_count": 0,
                                                                    "underestimate_count": 0})

        current["error_sum"] += abs_error
        current["count"] += 1

        if error > 0:
            current["underestimate_error_sum"] += abs_error
            current["underestimate_count"] += 1
        elif error < 0:
            current["overestimate_error_sum"] += abs_error
            current["overestimate_count"] += 1
        # if error == 0 discard

    ret = []
    for key, value in error_by_interval.items():
        mae = None
        overestimate_error = None
        underestimate_error = None

        if value["count"] != 0:
            mae = value["error_sum"] / value["count"]

        if value["overestimate_count"] != 0:
            overestimate_error = value["overestimate_error_sum"] / value["overestimate_count"]

        if value["underestimate_count"] != 0:
            underestimate_error = value["underestimate_error_sum"] / value["underestimate_count"]

        label = f"[{key}-{key + age_interval_width - 1}]"

        ret.append({"start_age": key,
                    "mae": mae,
                    "overestimate_mae": overestimate_error,
                    "underestimate_mae": underestimate_error,
                    "count": value["count"],
                    "overestimate_count": value["overestimate_count"],
                    "underestimate_count": value["underestimate_count"],
                    "age_interval_label": label,
                    })

    if verbose:
        print_error_by_age_intervals(ret)

    return ret


def evaluate(y, y_pred, verbose=True):
    """
    y and y_pred have to be list or numpy arrays
    """

    mae_int = np.mean(abs(np.rint(y) - np.rint(y_pred)))
    mae_float = np.mean(abs(y - y_pred))

    if verbose:
        print("")
        print(f"MAE int: {mae_int}")
        print(f"MAE float: {mae_float}")
        print("")

    return mae_int, mae_float


def evaluate_model(configuration_file_path):
    conf = configuration.read_configuration(configuration_file_path)

    model_path = conf["model_path"]

    preprocessing = conf["preprocessing"]
    eval_dict = conf["evaluate"]

    preprocessing_function_name = preprocessing["preprocessing_function_name"]
    enable_preprocessing = preprocessing["enabled"]

    test_set_path = eval_dict["test_set_path"]
    num_test_samples = eval_dict["num_test_samples"]

    batch_size = conf["batch_size"]

    save_predictions_dict = eval_dict["save_predictions"]

    save_predictions = save_predictions_dict["enabled"]
    save_predictions_path = save_predictions_dict["save_predictions_path"]

    predict_function_name = eval_dict["predict_function_name"]

    normalization_function_name = conf["normalization_function_name"]

    csv_path = conf["csv_path"]
    input_shape = conf["input_shape"]

    age_intervals_evaluation = eval_dict["age_intervals_evaluation"]

    evaluate_by_age_intervals_flag = age_intervals_evaluation["enabled"]
    age_interval_width = age_intervals_evaluation["age_interval_width"]

    if not enable_preprocessing:
        preprocessing_function = None
    elif preprocessing_function_name not in preprocessing_functions.AVAILABLE_PREPROCESSING_FUNCTIONS:
        raise Exception("The requested preprocessing function is not supported")
    else:
        preprocessing_function = preprocessing_functions.AVAILABLE_PREPROCESSING_FUNCTIONS[preprocessing_function_name]

    labels_dict = load_labels(csv_path, False)
    x_test, y_test = prepare_data_for_generator(test_set_path, labels_dict, num_test_samples)

    model = keras.models.load_model(model_path, custom_objects=CUSTOM_OBJECTS)

    predict_function = PREDICT_FUNCTIONS[predict_function_name]
    normalization_function = NORMALIZATION_FUNCTIONS[normalization_function_name]

    y_pred = predict_function(model, x_test, input_shape=input_shape, batch_size=batch_size,
                              preprocessing_function=preprocessing_function,
                              normalization_function=normalization_function)

    evaluate(y_test, y_pred, True)

    if evaluate_by_age_intervals_flag:
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
