import keras
import numpy as np
import h5py

class TrainDataGenerator(keras.utils.Sequence):
    def __init__(self, h5_file_path, num_samples, batch_size=32, normalization_function=None, shuffle=True):

        self.batch_size = batch_size
        self.h5_file_path = h5_file_path
        self.shuffle = shuffle
        self.num_samples = num_samples
        self.indices = np.arange(num_samples)
        self.normalization_function = normalization_function
        # this is called at initialization in order to create the indices for the subsequent data generation
        self.on_epoch_end()

    def __len__(self):
        """Determines the number of steps per epoch. Following
        https://keras.io/models/model/ advice on this parameter of fit_generator,
        it is defined as `ceil(len(samples) / batch_size)`"""
        return int(np.ceil(self.num_samples / self.batch_size))

    def __getitem__(self, index):
        """Generates one batch of data."""
        indices = self.indices[index * self.batch_size:(index + 1) * self.batch_size]

        with h5py.File(self.h5_file_path, "r") as h5_file:
            X = h5_file["X"]
            y = h5_file["Y"]
        # np.array creates a deep copy
            batch_X = np.array([X[i] for i in indices])
            batch_y = np.array([y[i] for i in indices])

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
    def __init__(self, X, batch_size=32, preprocessing_function=None, shuffle=True):
        self.X = X
        self.batch_size = batch_size
        self.preprocessing_function = preprocessing_function
        self.shuffle = shuffle
        self.indices = np.arange(len(self.X))
        # this is called at initialization in order to create the indices for the subsequent data generation
        self.on_epoch_end()

    def __len__(self):
        """Determines the number of steps per epoch. Following
        https://keras.io/models/model/ advice on this parameter of fit_generator,
        it is defined as `ceil(len(samples) / batch_size)`"""
        return int(np.ceil(len(self.X) / self.batch_size))

    def __getitem__(self, index):
        """Generates one batch of data."""
        indices = self.indices[index * self.batch_size:(index + 1) * self.batch_size]

        # np.array creates a deep copy
        batch_X = np.array([self.X[i] for i in indices])

        if self.preprocessing_function is not None:
            batch_X = self.preprocessing_function(batch_X)

        return batch_X

    def on_epoch_end(self):
        """Updates indices after each epoch. If shuffle was set to True, this also
        shuffles all the indices, in order to create different batches afterwards."""
        print("Updating indices...")
        if self.shuffle:
            np.random.shuffle(self.indices)