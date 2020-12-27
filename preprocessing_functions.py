from preprocessing import detect_from_image
from utilities.face_detector import FaceDetector
import cv2
import numpy as np


def standard_preprocessing(X, normalization_function=None):
    detector = FaceDetector()

    preprocessed_images = []

    for image in X:
        #detection
        detected_image = detect_from_image(image, detector)
        if detected_image is None:
            detected_image = image

        preprocessed_images.append(detected_image)

    numpy_preprocessed_images = np.array(preprocessed_images)

    # normalization
    if normalization_function is not None:
        return normalization_function(numpy_preprocessed_images)

    return numpy_preprocessed_images


AVAILABLE_PREPROCESSING_FUNCTIONS = {"standard_preprocessing_function": standard_preprocessing}