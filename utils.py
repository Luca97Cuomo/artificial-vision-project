import h5py
import os


def prepare_data_for_generator(data_path, labels_dict):
    identities = set(os.listdir(data_path)) & labels_dict.keys()
    image_paths = []
    labels = []
    for identity in identities:
        images = set(os.listdir(os.path.join(data_path, identity)))
        for image, age in labels_dict[identity].items():
            if image in images:
                image_paths.append(os.path.join(data_path, identity, image))
                labels.append(age)

    print(image_paths)

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