import argparse
import configuration
from models import NORMALIZATION_FUNCTIONS
from models import PREDICT_FUNCTIONS
from models import CUSTOM_OBJECTS
import preprocessing_functions
import keras
import numpy as np
from pathlib import Path
from utils import load_model


def prepare_data_to_predict(test_set_path):
    test_set_path = Path(test_set_path).resolve()
    identities = test_set_path.iterdir()

    image_paths = []
    count = 0

    for identity in identities:
        for image in identity.iterdir():
            image_paths.append(str(image.resolve()))
            count += 1

    print(f"The images found in the test set are {count}")

    return image_paths


def predict_model(configuration_file_path):
    conf = configuration.read_configuration(configuration_file_path)

    model_path = conf["model_path"]

    preprocessing = conf["preprocessing"]
    eval_dict = conf["evaluate"]

    preprocessing_function_name = preprocessing["preprocessing_function_name"]
    enable_preprocessing = preprocessing["enabled"]

    test_set_path = eval_dict["test_set_path"]

    batch_size = conf["batch_size"]

    save_predictions_dict = eval_dict["save_predictions"]

    save_predictions_path = save_predictions_dict["save_predictions_path"]

    predict_function_name = eval_dict["predict_function_name"]

    normalization_function_name = conf["normalization_function_name"]

    input_shape = conf["input_shape"]

    if not enable_preprocessing:
        preprocessing_function = None
    elif preprocessing_function_name not in preprocessing_functions.AVAILABLE_PREPROCESSING_FUNCTIONS:
        raise Exception("The requested preprocessing function is not supported")
    else:
        preprocessing_function = preprocessing_functions.AVAILABLE_PREPROCESSING_FUNCTIONS[preprocessing_function_name]

    x_test = prepare_data_to_predict(test_set_path)

    model = load_model(conf)

    predict_function = PREDICT_FUNCTIONS[predict_function_name]
    normalization_function = NORMALIZATION_FUNCTIONS[normalization_function_name]

    y_pred = predict_function(model, x_test, input_shape=input_shape, batch_size=batch_size,
                              preprocessing_function=preprocessing_function,
                              normalization_function=normalization_function)

    y_pred = np.rint(y_pred)

    print(f"Saving predictions to {save_predictions_path}")
    with open(save_predictions_path, 'w') as f:
        for i in range(len(x_test)):
            image_path = Path(x_test[i]).resolve()
            image = image_path.name
            identity = image_path.parent.name
            path = f"{identity}/{image}"
            f.write(f'{path},{int(round(y_pred[i]))}\r\n')
    print("Predictions saved")


def main():
    parser = argparse.ArgumentParser(description='Evaluate model')
    parser.add_argument('-c', '--configuration_file_path', type=str, help='The path of the configuration file',
                        required=True)

    args = parser.parse_args()

    predict_model(args.configuration_file_path)


if __name__ == '__main__':
    main()
