from utils import *
from preprocessing import load_labels
import json
import keras
import numpy as np
import preprocessing_functions
import argparse
from models import NORMALIZATION_FUNCTIONS
from models import PREDICT_FUNCTIONS


def evaluate(Y, Y_pred):
    mae_int = np.average(abs(np.rint(Y) - np.rint(Y_pred)))
    mae_float = np.average(abs(Y - Y_pred))
    return mae_int, mae_float


def evaluate_model(model_path, metadata_path, preprocessing_function, x_test, y_test, batch_size):
    model = keras.models.load_model(model_path)

    metadata = None
    with open(metadata_path) as json_file:
        metadata = json.load(json_file)

    predict_function_name = metadata["predict_function_name"]
    predict_function = PREDICT_FUNCTIONS[predict_function_name]
    input_shape = metadata["input_shape"]
    normalization_function_name = metadata["normalization_function_name"]
    normalization_function = NORMALIZATION_FUNCTIONS[normalization_function_name]

    y_pred = predict_function(model, x_test, input_shape=input_shape, batch_size=batch_size,
                              preprocessing_function=preprocessing_function,
                              normalization_function=normalization_function)

    mae_int, mae_float = evaluate(y_test, y_pred)

    print(f"MAE int: {mae_int}\nMAE float {mae_float}")


def main():
    parser = argparse.ArgumentParser(description='Train model')
    parser.add_argument('-model', '--model_path', type=str, help='The path of the model', required=True)
    parser.add_argument('-metadata', '--metadata_path', type=str, help='The path of the metadata file', required=True)
    parser.add_argument('-csv', '--csv_path', type=str, help='The path of the csv', required=True)
    parser.add_argument('-ts', '--test_set', type=str, help='The path of the test set', required=True)
    parser.add_argument('-nts', '--num_test_samples', type=int, help='The number of the test samples', required=True)
    parser.add_argument('-p', '--preprocessing_function_name', type=str,
                        help='The name of the preprocessing function that have to be used in order to preprocess the data.'
                        'The preprocessing function should apply the same preprocessing applied to the data in the training phase',
                        required=False, default=None)
    parser.add_argument('-v', '--verbose', action='store_true', help='Verbose')
    parser.add_argument('-b', '--batch_size', type=int, help='batch size', required=True)

    args = parser.parse_args()

    if args.preprocessing_function_name is None:
        preprocessing_function = None
    elif args.preprocessing_function_name not in preprocessing_functions.AVAILABLE_PREPROCESSING_FUNCTIONS:
        raise Exception("The requested preprocessing function is not supported")
    else:
        preprocessing_function = preprocessing_functions.AVAILABLE_PREPROCESSING_FUNCTIONS[args.preprocessing_function_name]

    labels_dict = load_labels(args.csv_path, False)
    x_test, y_test = prepare_data_for_generator(args.test_set, labels_dict, args.num_test_samples)

    evaluate_model(args.model_path, args.metadata_path, preprocessing_function, x_test, y_test, args.batch_size)


if __name__ == '__main__':
    main()
