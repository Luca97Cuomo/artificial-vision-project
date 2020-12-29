import argparse
import random
import cv2
import h5py
from utilities.face_detector import *
from pathlib import Path
from tqdm import tqdm
import shutil

# from imutils.face_utils import FaceAligner
# import dlib

DATASET_DIR = ''
LABEL_DIR = ''
random.seed(42)

"""
def pose_normalization_image(shape_predictor, image_path, new_image_path, verbose=False):
    # initialize dlib's face detector (HOG-based) and then create
    # the facial landmark predictor and the face aligner
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(shape_predictor)
    face_aligner = FaceAligner(predictor, desiredFaceWidth=192, desiredLeftEye=(0.30, 0.30))

    # load image
    image = cv2.imread(image_path)

    # convert it to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # detect faces in the grayscale image
    aligned_image = None
    rects = detector(gray, 2)
    if rects is not None and len(rects) > 0:
      # I am interested only in the first image of the dataset
      rect = rects[0]
      # align the face using facial landmarks
      aligned_image = face_aligner.align(image, gray, rect)

    if aligned_image is None:
        if verbose:
            print("Unable to align the image. Saving the original image")
        cv2.imwrite(new_image_path, image)
    else:
        if verbose:
            print("Successfully aligned the image. Saving the aligned image")
        cv2.imwrite(new_image_path, aligned_image)
"""


def detect_from_image(image, detector):
    height, width, channels = image.shape
    faces = detector.detect(image)
    if len(faces) <= 0:
        return None
    face = findRelevantFace(faces, width, height)

    rect = enclosing_square(face['roi'])
    cropped_image = cut(image, rect)

    return cropped_image


def detect_from_path(image_path, detector):
    image = cv2.imread(image_path)

    detected_image = detect_from_image(image, detector)
    if detected_image is None:
        print("Could not find faces for " + image_path)
        return image

    return detected_image


def remove_not_labeled_identities(identities, labels):
    new_identities = []
    removed_identities = 0
    for identity in identities:
        basename_identity = identity.name
        if basename_identity in labels:
            new_identities.append(identity)
        else:
            removed_identities += 1

    print(f"{removed_identities} removed identities")

    return new_identities


def generate_cropped_train_test_val(dataset_path, destination_path, val_fraction, test_fraction, detector,
                                    total_samples, labels):
    destination = Path(destination_path).resolve()
    dataset = Path(dataset_path).resolve()
    dirs = list(dataset.iterdir())

    dirs = remove_not_labeled_identities(dirs, labels)
    ##RIMUOVERE LE IMMAGINI CHE NON SONO ASSOCIATE A LABELS

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
                    cropped_image = detect_from_path(str(current_image), detector)
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


def count_dataset_images(dataset_path):
    identities = list(dataset_path.iterdir())
    number_of_images = 0

    for identity in identities:
        images = list(identity.iterdir())
        for image in images:
            number_of_images += 1

    return number_of_images


def resize_dataset(dataset_path, output_path, image_width, image_height):
    output_path = Path(output_path).resolve()
    dataset_path = Path(dataset_path).resolve()
    number_of_images = count_dataset_images(dataset_path)
    print(f"The images to resize are {number_of_images}")
    identities = list(dataset_path.iterdir())

    with tqdm(total=number_of_images) as pbar:
        for identity in identities:
            identity_path = output_path / identity.name
            identity_path.mkdir(exist_ok=True)
            images = list(identity.iterdir())
            for image in images:
                raw_image = cv2.imread(str(image))
                resized_raw_image = cv2.resize(raw_image, (image_width, image_height), interpolation=cv2.INTER_AREA)
                cv2.imwrite(str(identity_path / image.name), resized_raw_image)
                pbar.update(1)


def extract_virgin_dataset(virgin_dataset_path, dataset_to_extract_path, output_path):
    output_path = Path(output_path).resolve()
    virgin_dataset_path = Path(virgin_dataset_path).resolve()
    dataset_path = Path(dataset_to_extract_path).resolve()
    number_of_images = count_dataset_images(dataset_path)
    print(f"The images to resize are {number_of_images}")
    identities = list(dataset_path.iterdir())

    with tqdm(total=number_of_images) as pbar:
        for identity in identities:
            identity_destination_path = output_path / identity.name
            identity_destination_path.mkdir(exist_ok=True)
            images = list(identity.iterdir())
            for image in images:
                shutil.copy(str(virgin_dataset_path / identity.name / image.name),
                            str(identity_destination_path / image.name))
                pbar.update(1)


def generate_h5_dataset(datasets_path, output_path, dataset_name, labels, image_width, image_height):
    output_path = Path(output_path).resolve()
    destination_path = output_path / dataset_name
    images_not_in_csv = 0

    # the folder that contains cropped test, validation and training sets
    datasets_path = Path(datasets_path).resolve()

    dataset_path = datasets_path / dataset_name

    number_of_images = count_dataset_images(dataset_path)
    print(f"The images in {dataset_name} are {number_of_images}")

    with h5py.File(str(destination_path) + ".h5", 'w') as f:
        dataset_x = f.create_dataset("X", (number_of_images, image_width, image_height, 3), dtype='u1')
        dataset_y = f.create_dataset("Y", (number_of_images,), dtype="f")

        identities = list(dataset_path.iterdir())
        i = 0
        with tqdm(total=number_of_images) as pbar:
            for identity in identities:
                basename_identity = identity.name
                identity_labels = labels[basename_identity]
                images = list(identity.iterdir())

                for image in images:
                    basename_image = image.name
                    raw_image = cv2.imread(str(image))
                    resized_raw_image = cv2.resize(raw_image, (image_width, image_height), interpolation=cv2.INTER_AREA)
                    if basename_image in identity_labels:
                        dataset_x[i] = resized_raw_image
                        dataset_y[i] = identity_labels[basename_image]
                    else:
                        print(
                            f"The image {basename_image} for the identity {str(identity)} was found in the files of the dataset but not in the csv")
                        pbar.update(1)
                        images_not_in_csv += 1
                        continue
                    i += 1
                    pbar.update(1)

                print(f"{images_not_in_csv} not found in the csv")


def main():
    parser = argparse.ArgumentParser(description='Split dataset in training, validation and test sets.')
    parser.add_argument('-d', '--dataset_path', type=str, help='The absolute path of the dataset', required=True)
    parser.add_argument('-l', '--label_path', type=str, help='The absolute path of the file containing the labels',
                        required=True)
    parser.add_argument('-o', '--destination_path', type=str,
                        help='The absolute path of the folder that will contain the train, test and validation sets',
                        required=True)
    parser.add_argument('-R', '--round', action='store_true', help='Round the ages to the nearest integer')
    parser.add_argument('--h5_output_path', type=str,
                        help='The absolute path of the folder that will contain the h5 dataset files',
                        required=True)
    parser.add_argument('-n', '--number_of_images', type=int,
                        help='The number of images that will be taken from the dataset',
                        required=True)
    parser.add_argument('-v', '--val_fraction', type=float,
                        help='The fraction of the dataset that will be used for the validation set',
                        required=False, default=0.3)
    parser.add_argument('-t', '--test_fraction', type=float,
                        help='The fraction of the dataset that will be used for the test set',
                        required=False, default=0.3)

    args = parser.parse_args()

    val_fraction = args.val_fraction
    if args.val_fraction > 0.3:
        print(f"the validation fraction can't be more than 0.3, using 0.3")
        val_fraction = 0.3

    test_fraction = args.test_fraction
    if args.test_fraction > 0.3:
        print(f"the test fraction can't be more than 0.3, using 0.3")
        test_fraction = 0.3

    labels = load_labels(args.label_path, args.round)

    detector = FaceDetector()
    generate_cropped_train_test_val(args.dataset_path, args.destination_path, val_fraction, test_fraction, detector,
                                    args.number_of_images, labels)

    generate_h5_dataset(datasets_path=args.destination_path, output_path=args.h5_output_path, dataset_name="test_set",
                        labels=labels, image_width=224, image_height=224)

    generate_h5_dataset(datasets_path=args.destination_path, output_path=args.h5_output_path,
                        dataset_name="validation_set",
                        labels=labels, image_width=224, image_height=224)
    generate_h5_dataset(datasets_path=args.destination_path, output_path=args.h5_output_path,
                        dataset_name="training_set",
                        labels=labels, image_width=224, image_height=224)


if __name__ == '__main__':
    extract_virgin_dataset(r"C:\Users\utente\Desktop\Magistrale\magistrale\Visione artificiale\vggface2_train\train",
                           r"C:\Users\utente\Desktop\Magistrale\magistrale\Visione artificiale\cropped_dataset\test_set",
                           r"C:\Users\utente\Desktop\Magistrale\magistrale\Visione artificiale\virgin_datasets\virgin_test_set")
