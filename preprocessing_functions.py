from preprocessing import detect_from_image
from utilities.face_detector import FaceDetector
import cv2
import numpy as np


def standard_preprocessing(X, input_shape, normalization_function=None):
    detector = FaceDetector()

    preprocessed_images = []

    for image in X:
        #detection
        detected_image = detect_from_image(image, detector)
        if detected_image is None:
            detected_image = image

        image_width = input_shape[0]
        image_height = input_shape[1]

        # resize
        resized_image = cv2.resize(detected_image, (image_width, image_height), interpolation=cv2.INTER_AREA)

        preprocessed_images.append(resized_image)

    numpy_preprocessed_images = np.array(preprocessed_images)

    # normalization
    if normalization_function is not None:
        return normalization_function(numpy_preprocessed_images)

    return numpy_preprocessed_images


AVAILABLE_PREPROCESSING_FUNCTIONS = {"standard_preprocessing_function": standard_preprocessing}