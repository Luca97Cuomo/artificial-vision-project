from preprocessing import detect_from_image, detect_face_box_from_image
from utilities.face_detector import FaceDetector
import cv2
import numpy as np


def standard_preprocessing(x, input_shape):
    # preprocessing pipeline
    # 1) face detection
    # 2) resize

    detector = FaceDetector()

    preprocessed_images = []
    rect = None

    for image in x:
        #detection
        detected_image = detect_from_image(image, detector)
        if detected_image is None:
            detected_image = image

        # resize
        resized_image = cv2.resize(detected_image, (input_shape[0], input_shape[1]), interpolation=cv2.INTER_AREA)

        preprocessed_images.append(resized_image)

    numpy_preprocessed_images = np.array(preprocessed_images)
    return numpy_preprocessed_images

def demo_preprocessing(x, input_shape):
    
    detector = FaceDetector()
    detected_image, box = detect_face_box_from_image(x, detector)
    if detected_image is None:
        detected_image = x

    # resize
    resized_image = cv2.resize(detected_image, (input_shape[0], input_shape[1]), interpolation=cv2.INTER_AREA)
    batch = []
    batch.append(resized_image)
    return np.array(batch), box

AVAILABLE_PREPROCESSING_FUNCTIONS = {"standard_preprocessing_function": standard_preprocessing, "demo_preprocessing": demo_preprocessing}