import cv2
from pathlib import Path
import tensorflow as tf
import time
import os
from augmentations.augmentation import *
from random_bins import binner
from models import from_random_bins_classification_to_regression

ESC_KEY = 27
SPACE_KEY = 32
ONE_KEY = 49
TWO_KEY = 50
THREE_KEY = 51
FOUR_KEY = 52
FIVE_KEY = 53
SIX_KEY = 54
SEVEN_KEY = 55
RESET_KEY = 114


horizontal_augmenter = HorizontalMotionBlurAugmentation(probability=0)
vertical_augmenter = VerticalMotionBlurAugmentation(horizontal_augmenter, probability=0)
pixelate_augmenter = PixelateAugmentation(vertical_augmenter, probability=0)
gaussian_augmenter = GaussianNoiseAugmentation(pixelate_augmenter, probability=0)
brightness_augmenter = BrightnessAugmentation(gaussian_augmenter, probability=0)
final_augmenter = ContrastAugmentation(brightness_augmenter, probability=0)

AUGMENTERS = {
    ONE_KEY: [horizontal_augmenter, [1, 2, 3], 0],
    TWO_KEY: [vertical_augmenter, [1, 2, 3], 0],
    THREE_KEY: [pixelate_augmenter, [1, 2, 3], 0],
    FOUR_KEY: [gaussian_augmenter, [1, 2, 3, 4, 5], 0],
    FIVE_KEY: [brightness_augmenter, [-5, -4, -3, -2, -1, 1, 2, 3, 4, 5], 0],
    SIX_KEY: [final_augmenter, [-5, -4, -3, -2, -1, 1, 2, 3, 4, 5], 0]
}

NUMBER_OF_RVC_CLASSES = 101
BINNER = binner.Binner(n_classes=NUMBER_OF_RVC_CLASSES)


def regression_predict_demo(model, x, verbose=0):
    y = np.rint(model.predict(x, verbose=verbose))

    # do not round to int
    y = np.reshape(y, -1)
    return y.astype('int64')


def rvc_predict_demo(model, x, verbose=0):
    y = model.predict(x, verbose=verbose)

    y_processed = tf.map_fn(lambda element: tf.math.argmax(element), y, dtype=tf.dtypes.int64)

    if tf.__version__[0] == "2":
        y = y_processed.numpy()
    else:
        with tf.Session().as_default():
            y = y_processed.eval()

    return y.astype('int64')


def random_bins_classification_predict_demo(model, x, verbose=0):
    y = model.predict(x, verbose=verbose)

    if len(x) != len(y[0]):
        raise Exception("Unexpected number of predictions")

    y = np.rint(from_random_bins_classification_to_regression(y, BINNER))
    return y.astype('int64')


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
                augmented_faces = []
                for face in cropped_faces:
                    face = final_augmenter(face)
                    augmented_faces.append(face)
                cropped_faces = normalization_function(np.array(augmented_faces))
                y_pred = predict_function(model, cropped_faces, verbose)

                for prediction, box in zip(y_pred, boxes):
                    start_point = (box[0], box[1])
                    end_point = (box[0] + box[2], box[1] + box[3])
                    frame = final_augmenter(frame)
                    frame = cv2.rectangle(frame, start_point, end_point, color=(0, 255, 0), thickness=2)
                    cv2.putText(frame, str(prediction), start_point, font, fontScale=1, color=(0, 255, 0), thickness=2)

            post = time.perf_counter()
            loop_time = post - pre
            fps = int(round(1 / loop_time))
            cv2.putText(frame, f"FPS: {fps}", (15, 25), font, fontScale=1, color=(255, 255, 255), thickness=2)

            cv2.imshow("Age estimator", frame)

            if first_loop:
                print_demo_informations(save_predictions_path)
                first_loop = False

            k = cv2.waitKey(1)
            if k % 256 == ESC_KEY:
                # ESC pressed
                print("Bye :-)")
                break
            elif k % 256 == SPACE_KEY:
                # SPACE pressed
                space_key_function(save_predictions_path, frame, img_counter)
                img_counter += 1
            elif k % 256 == RESET_KEY:
                reset_augmentation()
            elif k % 256 in AUGMENTERS.keys():
                key_augmenter = AUGMENTERS[k]
                len_severity = len(key_augmenter[1])
                augmenter_severity_counter = key_augmenter[2]
                if augmenter_severity_counter == 0 and key_augmenter[0].probability == 0:
                    key_augmenter[0].probability = 1
                    key_augmenter[0].severity_values = [key_augmenter[1][0]]
                    augmenter_severity_counter = (augmenter_severity_counter + 1) % len_severity
                    key_augmenter[2] = augmenter_severity_counter

                elif augmenter_severity_counter == 0 and key_augmenter[0].probability == 1:
                    key_augmenter[0].probability = 0

                elif augmenter_severity_counter != 0:
                    key_augmenter[0].severity_values = [key_augmenter[1][augmenter_severity_counter]]
                    augmenter_severity_counter = (augmenter_severity_counter + 1) % len_severity
                    key_augmenter[2] = augmenter_severity_counter

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


def print_demo_informations(save_predictions_path):
    print("\nPress ESC to quit.\n")
    if save_predictions_path is not None:
        print(f"Press SPACE to save prediction in {save_predictions_path}.\n")

    print("How to use the demo:\n"
          "\n- Press 1 for horizontal motion blur\n"
          "\n- Press 2 for vertical motion blur\n"
          "\n- Press 3 for pixelate motion blur\n"
          "\n- Press 4 for gaussian noise corruption\n"
          "\n- Press 5 for brightness change corruption\n"
          "\n- Press 6 for contrast change corruption\n"
          "\n- Press R to reset all the corruptions\n")


def reset_augmentation():
    for aug in AUGMENTERS.values():
        aug[0].probability = 0
        aug[2] = 0


def space_key_function(save_predictions_path, frame, img_counter):
    if save_predictions_path is not None:
        img_name = f"opencv_frame_{img_counter}.png"
        save_path = Path(save_predictions_path).resolve() / img_name
        cv2.imwrite(str(save_path), frame)
        print(f"{save_path} written!")
    else:
        print("You must specify save_prediction_path in the configuration file.")


PREDICT_FUNCTIONS_DEMO = {"regression_predict_function": regression_predict_demo,
                          "rvc_predict_function": rvc_predict_demo,
                          "random_bins_classification_predict_function": random_bins_classification_predict_demo}
