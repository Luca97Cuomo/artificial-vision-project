import os
import argparse
import random
import shutil
import cv2
from utilities.face_detector import *

# from PIL import Image
# from IPython.display import display
# import numpy

DATASET_DIR = ''
LABEL_DIR = ''
random.seed(42)


def detect(image_path, detector):
    image = cv2.imread(image_path)
    height, width, channels = image.shape
    faces = detector.detect(image)
    if len(faces) <= 0:
        print("Non ho trovato facce per " + image_path)
        return image
    face = findRelevantFace(faces, width, height)

    rect = enclosing_square(face['roi'])
    cropped_image = cut(image, rect)

    return cropped_image


def generate_cropped_train_test_val(dataset_path, destination_path, val_fraction, test_fraction, detector,
                                    total_samples):
    dirs = os.listdir(dataset_path)

    """
    Conto il numero totale di immagini presenti nel dataset_path
    files = []
    for current_dir in dirs:
        current_dir = os.path.join(dataset_path, current_dir)
        files = files + os.listdir(current_dir)

    total_samples = len(files)
    
    """

    val_samples = round(total_samples * val_fraction)
    test_samples = round(total_samples * test_fraction)
    training_samples = total_samples - test_samples - val_samples

    print("training samples should be " + str(training_samples))
    print("validation samples should be " + str(val_samples))
    print("test samples should be " + str(test_samples))
    random.shuffle(dirs)

    samples_evaluated = 0
    true_validation_samples = 0
    true_test_samples = 0
    true_training_samples = 0

    try:
        os.mkdir(os.path.join(destination_path, "training_set"))
        os.mkdir(os.path.join(destination_path, "validation_set"))
        os.mkdir(os.path.join(destination_path, "test_set"))
    except OSError:
        print("Creation of the directories failed")

    for entity in dirs:
        complete_current_dir = os.path.join(dataset_path, entity)
        current_images = os.listdir(complete_current_dir)
        if val_samples > 0:
            val_samples = val_samples - len(current_images)
            os.mkdir(os.path.join(destination_path, "validation_set", entity))
            for image in current_images:
                true_validation_samples += 1
                samples_evaluated += 1
                if (samples_evaluated % 1000) == 0:
                    print("Evaluated " + str(samples_evaluated) + "images")
                cropped_image = detect(os.path.join(complete_current_dir, image), detector)
                cv2.imwrite(os.path.join(destination_path, "validation_set", entity, image), cropped_image)
        elif test_samples > 0:
            test_samples = test_samples - len(current_images)
            os.mkdir(os.path.join(destination_path, "test_set", entity))
            for image in current_images:
                true_test_samples += 1
                samples_evaluated += 1
                if (samples_evaluated % 1000) == 0:
                    print("Evaluated " + str(samples_evaluated) + "images")
                cropped_image = detect(os.path.join(complete_current_dir, image), detector)
                cv2.imwrite(os.path.join(destination_path, "test_set", entity, image), cropped_image)
        elif training_samples > 0:
            training_samples = training_samples - len(current_images)
            os.mkdir(os.path.join(destination_path, "training_set", entity))
            for image in current_images:
                true_training_samples += 1
                samples_evaluated += 1
                if (samples_evaluated % 1000) == 0:
                    print("Evaluated " + str(samples_evaluated) + "images")
                cropped_image = detect(os.path.join(complete_current_dir, image), detector)
                cv2.imwrite(os.path.join(destination_path, "training_set", entity, image), cropped_image)
        else:
            break

    print("training samples are " + str(true_training_samples))
    print("validation samples are " + str(true_validation_samples))
    print("test samples are " + str(true_test_samples))


def load_labels(labels_path, rounded):
    with open(labels_path, 'r') as f:
        lines = f.readlines()

    labels = {}
    for line in lines:
        identity, image_age = line.split(sep="/")
        image, age = image_age.split(sep=",")
        age = age.strip("\n")

        if rounded:
            age = round(float(age))
        else:
            age = round(float(age), 3)

        if identity not in labels.keys():
            labels[identity] = {image: age}
        else:
            labels[identity][image] = age

    return labels


def main():
    parser = argparse.ArgumentParser(description='Split dataset in training, validation and test sets.')
    parser.add_argument('dataset_path', type=str, help='The absolute path of the dataset')
    # parser.add_argument('label_path', type=str, help='The absolute path of the file containing the labels')
    parser.add_argument('destination_path', type=str,
                        help='The absolute path of the folder that will contain the train, test and validation sets')
    # parser.add_argument('-R', '--round', action='store_true', help='Round the ages to the nearest integer')

    args = parser.parse_args()

    # labels = load_labels(args.label_path, args.round)
    detector = FaceDetector()
    generate_cropped_train_test_val(args.dataset_path, args.destination_path, 0.2, 0.2, detector, 1000)


if __name__ == '__main__':
    main()
