from demo.utils import *
import configuration
import preprocessing_functions
from models import NORMALIZATION_FUNCTIONS
import argparse
from utils import load_model


def estimate_age(conf_path, use_images=False):
    conf = configuration.read_configuration(conf_path)
    verbose = conf["verbose"]
    verbose = 1 if verbose is True else 0

    model_path = conf["model_path"]
    input_shape = conf["input_shape"]
    normalization_function_name = conf["normalization_function_name"]

    normalization_function = NORMALIZATION_FUNCTIONS[normalization_function_name]

    preprocessing_conf = conf["preprocessing"]
    enable_preprocessing = preprocessing_conf["enabled"]
    preprocessing_function_name = preprocessing_conf["preprocessing_function_name"]

    if not enable_preprocessing:
        preprocessing_function = None
    elif preprocessing_function_name not in preprocessing_functions.AVAILABLE_PREPROCESSING_FUNCTIONS:
        raise Exception("The requested preprocessing function is not supported")
    else:
        preprocessing_function = preprocessing_functions.AVAILABLE_PREPROCESSING_FUNCTIONS[preprocessing_function_name]

    evaluate_conf = conf["evaluate"]

    if evaluate_conf["save_predictions"]["enabled"]:
        save_predictions_path = evaluate_conf["save_predictions"]["save_predictions_path"]
    else:
        save_predictions_path = None

    if use_images:
        images_path = evaluate_conf["test_set_path"]
    else:
        images_path = None

    predict_function_name = evaluate_conf["predict_function_name"]
    predict_function = PREDICT_FUNCTIONS_DEMO[predict_function_name]

    model = load_model(conf)

    if verbose:
        print(f"Model input shape: {input_shape}")

    if use_images:
        predict_from_images(model, input_shape, images_path, save_predictions_path, preprocessing_function,
                            normalization_function, predict_function, verbose)
    else:
        predict_from_camera(model, input_shape, save_predictions_path, preprocessing_function,
                            normalization_function, predict_function)


def main():
    parser = argparse.ArgumentParser(description='Estimates your age. Use it only with TF2.')
    parser.add_argument('-c', '--configuration_path', type=str, help='The path of the configuration', required=True)
    parser.add_argument('-i', '--images', action="store_true",
                        help='Uses images instead of camera if provided. The path of images folder must be provided'
                             'in the configuration file using test_set_path in the evaluate section.')
    args = parser.parse_args()
    if tf.__version__[0] != '2':
        raise Exception("Tensorflow version is not supported. Use TF2.")

    conf = Path(args.configuration_path).resolve()
    # This lise solved a bug relate to opencl
    cv2.ocl.setUseOpenCL(False)

    estimate_age(str(conf), use_images=args.images)


if __name__ == '__main__':
    main()
