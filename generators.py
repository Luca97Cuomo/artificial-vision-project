import keras
import numpy as np
from skimage.io import imread
import cv2


class TrainDataGenerator(keras.utils.Sequence):
    def __init__(self, data_paths, labels, input_shape, batch_size=32, normalization_function=None, shuffle=True):
        if len(data_paths) != len(labels):
            raise ValueError("Inappropriate sizes for the provided data. Check"
                             " that they are of the same size.")

        self.batch_size = batch_size
        self.labels = labels
        self.data_paths = data_paths
        self.input_shape = input_shape
        self.shuffle = shuffle
        self.indices = np.arange(len(self.X))
        self.normalization_function = normalization_function
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
        # also perform the resize of the image
        batch_X = np.array([cv2.resize(imread(self.data_paths[i], (self.input_shape[0], self.input_sahape[1]),
                                              interpolation=cv2.INTER_AREA)) for i in indices])

        batch_y = np.array([self.labels[i] for i in indices])

        if self.normalization_function is not None:
            batch_X = self.normalization_function(batch_X)

        return batch_X, batch_y

    def on_epoch_end(self):
        """Updates indices after each epoch. If shuffle was set to True, this also
        shuffles all the indices, in order to create different batches afterwards."""
        print("Updating indices...")
        if self.shuffle:
            np.random.shuffle(self.indices)


class PredictDataGenerator(keras.utils.Sequence):
    def __init__(self, data_paths, input_shape, batch_size=32, preprocessing_function=None, normalization_function=None, shuffle=True):
        self.data_paths = data_paths
        self.input_shape = input_shape
        self.batch_size = batch_size
        self.preprocessing_function = preprocessing_function
        self.normalization_function = normalization_function
        self.shuffle = shuffle
        self.indices = np.arange(len(self.X))
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
        # also perform the resize of the image
        batch_X = np.array([cv2.resize(imread(self.data_paths[i], (self.input_shape[0], self.input_sahape[1]),
                                              interpolation=cv2.INTER_AREA)) for i in indices])

        if self.preprocessing_function is not None:
            batch_X = self.preprocessing_function(batch_X, normalization_function=self.normalization_function)

        return batch_X

    def on_epoch_end(self):
        """Updates indices after each epoch. If shuffle was set to True, this also
        shuffles all the indices, in order to create different batches afterwards."""
        print("Updating indices...")
        if self.shuffle:
            np.random.shuffle(self.indices)