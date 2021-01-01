from preprocessing import detect_from_image
from utilities.face_detector import FaceDetector
import cv2
import numpy as np


def standard_preprocessing(x, input_shape, return_rect=False):
    # preprocessing pipeline
    # 1) face detection
    # 2) resize

    detector = FaceDetector()

    preprocessed_images = []
    rect = None

    for image in x:
        #detection
        detected_image, rect = detect_from_image(image, detector)
        if detected_image is None:
            detected_image = image
            rect = None

        # resize
        resized_image = cv2.resize(detected_image, (input_shape[0], input_shape[1]), interpolation=cv2.INTER_AREA)

        preprocessed_images.append(resized_image)

    numpy_preprocessed_images = np.array(preprocessed_images)
    if return_rect:
        return numpy_preprocessed_images, rect
    else:
        return numpy_preprocessed_images


AVAILABLE_PREPROCESSING_FUNCTIONS = {"standard_preprocessing_function": standard_preprocessing}
