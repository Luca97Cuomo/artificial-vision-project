from preprocessing import detect_from_image, detect_relevant_face_box_from_image, detect_faces
from utilities.face_detector import FaceDetector
import cv2
import numpy as np

detector = FaceDetector()


def standard_preprocessing(x, input_shape):
    # preprocessing pipeline
    # 1) face detection
    # 2) resize

    preprocessed_images = []
    rect = None

    for image in x:
        detected_image = detect_from_image(image, detector)
        if detected_image is None:
            detected_image = image

        # resize
        resized_image = cv2.resize(detected_image, (input_shape[0], input_shape[1]), interpolation=cv2.INTER_AREA)

        preprocessed_images.append(resized_image)

    numpy_preprocessed_images = np.array(preprocessed_images)
    return numpy_preprocessed_images


def demo_preprocessing(x, input_shape):
    batch = []
    boxes = []

    for detected_image, box in detect_faces(x, detector):
        resized_image = cv2.resize(detected_image, (input_shape[0], input_shape[1]), interpolation=cv2.INTER_AREA)
        batch.append(resized_image)
        boxes.append(box)
    if len(batch) == 0:
        batch.append(cv2.resize(x, (input_shape[0], input_shape[1]), interpolation=cv2.INTER_AREA))

        boxes = None

    return np.array(batch), boxes

AVAILABLE_PREPROCESSING_FUNCTIONS = {"standard_preprocessing_function": standard_preprocessing, "demo_preprocessing": demo_preprocessing}