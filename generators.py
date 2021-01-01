import keras
import numpy as np
from numpy.random import RandomState
import cv2
from augmentations import augmentation


class DataGenerator(keras.utils.Sequence):
    def __init__(self, data_paths, labels, input_shape, batch_size=32, preprocessing_function=None,
                 normalization_function=None, shuffle=True, random_seed=42, augmenter=None):
        """
        If labels is None it means that the generator has to be used in predict mode where the labels are not required
        if augmentation is not desired an instance of NullAugmentation should be passed
        """
        self.data_paths = data_paths
        self.labels = labels
        self.input_shape = input_shape
        self.batch_size = batch_size
        self.preprocessing_function = preprocessing_function
        self.normalization_function = normalization_function
        # if labels is None, the shuffle must not be performed in order to not lose correspondence with the input
        self.shuffle = shuffle
        self.randomness = RandomState(random_seed)

        if augmenter is None:
            augmenter = augmentation.NullAugmentation()

        self.augmenter = augmenter
        self.indices = np.arange(len(self.data_paths))
        # this is called at initialization in order to create the indices for the subsequent data generation
        self.on_epoch_end()

    def __len__(self):
        """Determines the number of steps per epoch. Following
        https://keras.io/models/model/ advice on this parameter of fit_generator,
        it is defined as `ceil(len(samples) / batch_size)`"""
        return int(np.ceil(len(self.data_paths) / self.batch_size))

    def __getitem__(self, index):
        """Generates one batch of data."""
        indices = self.indices[index * self.batch_size:(index + 1) * self.batch_size]

        # np.array creates a deep copy
        batch_x = np.array([self.augmenter(cv2.imread(self.data_paths[i])) for i in indices])

        if self.preprocessing_function is not None:
            batch_x = self.preprocessing_function(batch_x, self.input_shape)

        if self.normalization_function is not None:
            batch_x = self.normalization_function(batch_x)

        if self.labels is None:
            return batch_x

        batch_y = np.array([self.labels[i] for i in indices])

        return batch_x, batch_y

    def on_epoch_end(self):
        """Updates indices after each epoch. If shuffle was set to True, this also
        shuffles all the indices, in order to create different batches afterwards."""
        print("Updating indices...")
        if self.shuffle and self.labels is not None:
            self.randomness.shuffle(self.indices)
