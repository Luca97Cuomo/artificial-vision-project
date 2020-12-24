import os
import argparse
import random
import shutil
import cv2
from utilities.face_detector import *
from pathlib import Path
from tqdm import tqdm

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
        print("Could not find faces for " + image_path)
        return image
    face = findRelevantFace(faces, width, height)

    rect = enclosing_square(face['roi'])
    cropped_image = cut(image, rect)

    return cropped_image


def generate_cropped_train_test_val(dataset_path, destination_path, val_fraction, test_fraction, detector,
                                    total_samples):
    destination = Path(destination_path).resolve()
    dataset = Path(dataset_path).resolve()
    dirs = list(dataset.iterdir())

    total_len_dirs = len(dirs)
    val_len_dirs = round(total_len_dirs * val_fraction)
    test_len_dirs = round(total_len_dirs * test_fraction)

    random.shuffle(dirs)
    test_dirs = dirs[:test_len_dirs]
    val_dirs = dirs[test_len_dirs:test_len_dirs + val_len_dirs]
    training_dirs = dirs[test_len_dirs + val_len_dirs:]

    val_samples = round(total_samples * val_fraction)
    test_samples = round(total_samples * test_fraction)
    training_samples = total_samples - test_samples - val_samples

    print("training samples should be " + str(training_samples))
    print("validation samples should be " + str(val_samples))
    print("test samples should be " + str(test_samples))

    training_set_path = destination / "training_set"
    test_set_path = destination / "test_set"
    validation_set_path = destination / "validation_set"

    training_set_path.mkdir(parents=True, exist_ok=True)
    test_set_path.mkdir(exist_ok=True)
    validation_set_path.mkdir(exist_ok=True)

    extract_cropped_subset(test_set_path, detector, test_dirs, test_samples)
    extract_cropped_subset(validation_set_path, detector, val_dirs, val_samples)
    extract_cropped_subset(training_set_path, detector, training_dirs, training_samples)


def extract_cropped_subset(destination_path, detector, dirs, total_samples):
    evaluated_samples = 0
    i = 0

    # iterdir() could return the files in different orders, in order to avoid this, a list of images in
    # a specific order is saved in memory
    identity_images = {}
    for identity in dirs:
        identity_images[identity] = list(identity.iterdir())

    with tqdm(total=total_samples) as pbar:
        while evaluated_samples < total_samples:
            prev_evaluated_samples = evaluated_samples
            for identity, images in identity_images.items():
                try:
                    current_image = images[i]
                except IndexError:
                    print(f"Directory {identity.name} exhausted")
                else:
                    identity_path = destination_path / identity.name
                    identity_path.mkdir(exist_ok=True)
                    cropped_image = detect(str(current_image), detector)
                    cv2.imwrite(str(identity_path / current_image.name), cropped_image)
                    evaluated_samples += 1
                    pbar.update(1)
                    if evaluated_samples >= total_samples:
                        break

            i += 1
            if prev_evaluated_samples == evaluated_samples:
                print("Directories exhausted before getting all the samples")
                break

    print(f"For {destination_path.name} evaluated {evaluated_samples} images")


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
    parser.add_argument('-d', '--dataset_path', type=str, help='The absolute path of the dataset', required=True)
    parser.add_argument('-l', '--label_path', type=str, help='The absolute path of the file containing the labels',
                        required=True)
    parser.add_argument('-o', '--destination_path', type=str,
                        help='The absolute path of the folder that will contain the train, test and validation sets',
                        required=True)
    parser.add_argument('-R', '--round', action='store_true', help='Round the ages to the nearest integer')

    args = parser.parse_args()

    labels = load_labels(args.label_path, args.round)

    print(labels["n000002"])
    detector = FaceDetector()
    generate_cropped_train_test_val(args.dataset_path, args.destination_path, 0.2, 0.2, detector, 10000)


if __name__ == '__main__':
    main()
