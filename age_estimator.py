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


def regression_predict_demo(model, x):

    y = model.predict(x, verbose=1)

    # do not round to int
    return np.reshape(y, -1)


def rvc_predict_demo(model, x):

    y = model.predict(x, verbose=1)

    y_processed = tf.map_fn(lambda element: tf.math.argmax(element), y, dtype=tf.dtypes.int64)

    return y_processed.numpy()


PREDICT_FUNCTIONS_DEMO = {"regression_predict_function": regression_predict_demo, "rvc_predict_function": rvc_predict_demo}


def estimate_age(conf_path):
    conf = configuration.read_configuration(conf_path)
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

    predict_function_name = evaluate_conf["predict_function_name"]
    predict_function = PREDICT_FUNCTIONS_DEMO[predict_function_name]

    model = load_model(model_path, custom_objects=CUSTOM_OBJECTS)

    print(f"model input shape: {input_shape}")

    cam = cv2.VideoCapture(0)

    img_counter = 0

    while True:
        ret, frame = cam.read()
        if not ret:
            print("Failed capturing a frame")
            break

        if frame is not None:
            
            x, box = preprocessing_function(frame, input_shape)
            x = normalization_function(x)
            y_pred = predict_function(model, x)
            font = cv2.FONT_HERSHEY_SIMPLEX
            
            if box is not None:
                start_point = (box[0], box[1])
                end_point = (box[0] + box[2], box[1] + box[3])
                frame = cv2.rectangle(frame, start_point, end_point, color=(0, 255, 0), thickness=2)
                cv2.putText(frame, str(y_pred), start_point, font, fontScale=1, color=(0, 255, 0), thickness=2)
            
            cv2.imshow("Age estimator", frame)
            k = cv2.waitKey(1)
            if k % 256 == 27:
                # ESC pressed
                print("Bye :-)")
                break
            elif k % 256 == 32:
                # SPACE pressed
                img_name = f"opencv_frame_{img_counter}.png"
                save_path = Path(save_predictions_path).resolve() / img_name
                cv2.imwrite(str(save_path), frame)
                print(f"{save_path.name} written!")
                img_counter += 1

    cam.release()

    cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description='Estimates your age.')
    parser.add_argument('-c', '--configuration_path', type=str, help='The path of the configuration',
                        required=True)
    args = parser.parse_args()

    conf = Path(args.configuration_path).resolve()
    estimate_age(str(conf))


if __name__ == '__main__':
    main()
