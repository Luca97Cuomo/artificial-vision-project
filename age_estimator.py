from demo.utils import *
import configuration
import preprocessing_functions
from models import NORMALIZATION_FUNCTIONS, AVAILABLE_OUTPUT_TYPES
import argparse
from utils import load_model


def estimate_age(conf_path, use_images=False):
    conf = configuration.read_configuration(conf_path)
    verbose = conf["verbose"]
    verbose = 1 if verbose is True else 0

    model_path = conf["model_path"]
    input_shape = conf["input_shape"]
    backend_name = conf["build"]["backend"]["name"]

    normalization_function_name = backend_name + "_normalization"
    if normalization_function_name not in NORMALIZATION_FUNCTIONS:
        raise Exception("The normalization function is not available")
    normalization_function = NORMALIZATION_FUNCTIONS[normalization_function_name]

    preprocessing_conf = conf["preprocessing"]
    preprocessing_function_name = preprocessing_conf["preprocessing_function_name"]

    if preprocessing_function_name not in preprocessing_functions.AVAILABLE_PREPROCESSING_FUNCTIONS:
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

    output_type = conf["output_type"]
    if output_type not in AVAILABLE_OUTPUT_TYPES:
        raise Exception("The output type is not available")

    predict_function_name = output_type + "_predict_function"
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
    parser = argparse.ArgumentParser(description='Estimates your age.')
    parser.add_argument('-c', '--configuration_path', type=str, help='The path of the configuration', required=True)
    parser.add_argument('-i', '--images', action="store_true",
                        help='Uses images instead of camera if provided. The path of images folder must be provided'
                             'in the configuration file using test_set_path in the evaluate section.')
    args = parser.parse_args()

    conf = Path(args.configuration_path).resolve()
    # This lise solved a bug relate to opencl
    cv2.ocl.setUseOpenCL(False)

    estimate_age(str(conf), use_images=args.images)


if __name__ == '__main__':
    main()
