from keras.models import load_model
import cv2
import argparse
from pathlib import Path
import configuration
import preprocessing_functions
from models import NORMALIZATION_FUNCTIONS
from models import PREDICT_FUNCTIONS


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
    predict_function = PREDICT_FUNCTIONS[predict_function_name]

    model = load_model(model_path, compile=False)

    print(f"model input shape: {input_shape}")

    cam = cv2.VideoCapture(0)

    cv2.namedWindow("Age estimator")

    img_counter = 0

    while True:
        ret, frame = cam.read()
        if not ret:
            print("Failed capturing a frame")
            break

        if frame is not None:
            y_pred, box = predict_function(model, [frame], input_shape=input_shape, batch_size=1,
                                           preprocessing_function=preprocessing_function,
                                           normalization_function=normalization_function,
                                           return_rect=True)
            print(f"box= {box}")

            print(f"age = {y_pred}")

            # cv2.putText(frame, str(age), (), font, fontScale=1, color=(0, 255, 0), thickness=2)
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

    conf = Path(args.configuarion_path).resolve()
    estimate_age(str(conf))


if __name__ == '__main__':
    main()
