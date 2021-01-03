import h5py
import os
import numpy as np
from models import CUSTOM_OBJECTS
import tensorflow as tf
import keras


def prepare_data_for_generator(data_path, labels_dict, num_samples):
    # take only the identities that are both in the data_path and in the csv file
    identities = set(os.listdir(data_path)) & labels_dict.keys()
    identity_images = {}
    for identity in identities:
        identity_images[identity] = os.listdir(os.path.join(data_path, identity))

    image_paths = []
    labels = []
    count = 0
    i = 0

    while count < num_samples:
        prec = count
        for identity, images in identity_images.items():
            try:
                current_image = images[i]
            except IndexError:
                pass
                # print(f"Directory {identity} exhausted")
            else:
                # search age in the labels_dict
                if current_image in labels_dict[identity]:
                    age = labels_dict[identity][current_image]
                    image_paths.append(os.path.join(data_path, identity, current_image))
                    labels.append(age)
                    count += 1
                    if count >= num_samples:
                        break

        if prec == count:
            print("Directories exhausted before getting all the samples")
            break

        i += 1

    print(f"For {data_path} evaluated {count} images")

    return image_paths, labels


def read_dataset_h5(dataset_path, verbose):
    f = h5py.File(dataset_path, 'r')

    if verbose:
        print("Reading X...")
    X = f['X']
    if verbose:
        print("Reading Y...")
    Y = f['Y']

    if verbose:
        print(X.shape, Y.shape)

    return X, Y, f

    # Not close the file here
    # f.close()


def one_hot_encoded_labels(labels, number_of_classes):
    hot_encoded_labels = []
    for i in range(len(labels)):
        label = round(labels[i])
        formatted_label = np.zeros(number_of_classes)
        formatted_label[label] = 1

        hot_encoded_labels.append(formatted_label)

    return np.array(hot_encoded_labels)


def load_model(conf):
    tensorflow_version = conf["tf_version"]
    model_path = conf["model_path"]
    if tensorflow_version == 1:
        model = keras.models.load_model(model_path, custom_objects=CUSTOM_OBJECTS)
    elif tensorflow_version == 2:
        model = tf.keras.models.load_model(model_path, custom_objects=CUSTOM_OBJECTS)
    else:
        raise Exception("Tensorflow version not supported")
    return model
