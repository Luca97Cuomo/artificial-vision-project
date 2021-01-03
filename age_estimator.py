from tensorflow.keras.models import load_model
import cv2
import argparse
from pathlib import Path
import configuration
import preprocessing_functions
from models import NORMALIZATION_FUNCTIONS
from models import CUSTOM_OBJECTS
import numpy as np
import tensorflow as tf
import time
import os

ESC_KEY = 27
SPACE_KEY = 32


def regression_predict_demo(model, x, verbose=0):
    y = int(round((model.predict(x, verbose=verbose))))

    # do not round to int
    return np.reshape(y, -1)


def rvc_predict_demo(model, x, verbose=0):
    y = model.predict(x, verbose=verbose)

    y_processed = tf.map_fn(lambda element: tf.math.argmax(element), y, dtype=tf.dtypes.int64)

    # You need tensorflow 2.x.x to run this function
    return y_processed.numpy()


def predict_from_camera(model, input_shape, save_predictions_path, preprocessing_function, normalization_function,
                        predict_function, verbose=0):

    cam = cv2.VideoCapture(0)

    img_counter = 0
    font = cv2.FONT_HERSHEY_SIMPLEX
    first_loop = True

    while True:
        ret, frame = cam.read()
        pre = time.perf_counter()
        if not ret:
            print("Failed capturing a frame")
            break

        if frame is not None:

            cropped_faces, boxes = preprocessing_function(frame, input_shape)
            if cropped_faces is not None and boxes is not None:
                cropped_faces = normalization_function(cropped_faces)
                y_pred = predict_function(model, cropped_faces, verbose)

                for prediction, box in zip(y_pred, boxes):
                    start_point = (box[0], box[1])
                    end_point = (box[0] + box[2], box[1] + box[3])
                    frame = cv2.rectangle(frame, start_point, end_point, color=(0, 255, 0), thickness=2)
                    cv2.putText(frame, str(prediction), start_point, font, fontScale=1, color=(0, 255, 0), thickness=2)

            post = time.perf_counter()
            loop_time = post - pre
            fps = int(round(1 / loop_time))
            cv2.putText(frame, f"FPS: {fps}", (15, 25), font, fontScale=1, color=(255, 255, 255), thickness=2)

            cv2.imshow("Age estimator", frame)

            if first_loop:
                print("\nPress ESC to quit.\n")
                if save_predictions_path is not None:
                    print(f"Press SPACE to save prediction in {save_predictions_path}.\n")
                first_loop = False

            k = cv2.waitKey(1)
            if k % 256 == ESC_KEY:
                # ESC pressed
                print("Bye :-)")
                break
            elif k % 256 == SPACE_KEY:
                # SPACE pressed
                if save_predictions_path is not None:
                    img_name = f"opencv_frame_{img_counter}.png"
                    save_path = Path(save_predictions_path).resolve() / img_name
                    cv2.imwrite(str(save_path), frame)
                    print(f"{save_path} written!")
                    img_counter += 1
                else:
                    print("You must specify save_prediction_path in the configuration file.")

    cam.release()

    cv2.destroyAllWindows()


def predict_from_images(model, input_shape, images_path, save_predictions_path, preprocessing_function,
                        normalization_function, predict_function, verbose):

    font = cv2.FONT_HERSHEY_SIMPLEX

    images_path = str(Path(images_path).resolve())
    for image in os.listdir(images_path):
        frame = cv2.imread(os.path.join(images_path, image))

        if frame is not None:
            cropped_faces, boxes = preprocessing_function(frame, input_shape, True)
            if cropped_faces is not None and boxes is not None:
                cropped_faces = normalization_function(cropped_faces)
                y_pred = predict_function(model, cropped_faces, verbose)

                for prediction, box in zip(y_pred, boxes):
                    start_point = (box[0], box[1])
                    end_point = (box[0] + box[2], box[1] + box[3])
                    frame = cv2.rectangle(frame, start_point, end_point, color=(0, 255, 0), thickness=2)
                    cv2.putText(frame, str(prediction), start_point, font, fontScale=1, color=(0, 255, 0), thickness=2)
                if save_predictions_path is not None:
                    cv2.imwrite(str(Path(save_predictions_path).resolve() / image), frame)

                    if verbose:
                        print("Store result in " + str(Path(save_predictions_path).resolve() / image))
                else:
                    print("Cannot store predictions. You need to pass a suitable path in the configuration file.")


PREDICT_FUNCTIONS_DEMO = {"regression_predict_function": regression_predict_demo,
                          "rvc_predict_function": rvc_predict_demo}


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

    model = load_model(model_path, custom_objects=CUSTOM_OBJECTS)

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