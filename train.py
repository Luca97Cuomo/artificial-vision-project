from keras.callbacks import ModelCheckpoint, EarlyStopping, CSVLogger, ReduceLROnPlateau
import pickle
import numpy as np
import keras
import os
import json
import argparse
from utils import read_dataset


class DataGenerator(keras.utils.Sequence):
    def __init__(self, X, y, batch_size=32, shuffle=True):
        if len(X) != len(y):
            raise ValueError("Inappropriate sizes for the provided data. Check"
                             " that they are of the same size.")

        self.batch_size = batch_size
        self.y = y
        self.X = X
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

        batch_X = np.array([self.X[i] for i in indices])
        batch_y = np.array([self.y[i] for i in indices]) # maybe it depends on the network

        return batch_X, batch_y

    def on_epoch_end(self):
        """Updates indices after each epoch. If shuffle was set to True, this also
        shuffles all the indices, in order to create different batches afterwards."""
        print("Updating indices...")
        if self.shuffle:
            np.random.shuffle(self.indices)


def train_model(model_path, metafile_path, output_dir, batch_size, X_train, Y_train, X_val, Y_val, training_epochs, initial_epoch):
    model_name = os.path.basename(model_path)

    model = keras.models.load_model(model_path)

    metadata = None
    with open(metafile_path) as json_file:
        metadata = json.load(json_file)

    monitored_val_quantity = metadata["val_metric_name"]

    training_data_generator = DataGenerator(X_train, Y_train, batch_size=batch_size)
    validation_data_generator = DataGenerator(X_val, Y_val) # why not pass the batch size?

    append = None
    if initial_epoch == 0:
        append = False
    else:
        append = True
    logger_callback = CSVLogger(os.path.join(output_dir, model_name) + ".log", append=append) # remember to put it to True if continuing a training

    # divides learning rate by 5 if no improvement is found on validation over 3 epochs.
    # this is made in order to avoid stagnating too much: empirical evidence suggests that reducing learning rate when stagnating
    # can significantly help a model training process. This is probably because it can stagnate because of strange shapes
    # in the loss function which make it impossible for the optimizer to fall in better minima.
    validation_min_delta = 0.1 / 100  # minimum variation of the monitored quantity for it to be considered improved
    reduce_lr_plateau_callback = ReduceLROnPlateau(monitor=monitored_val_quantity, mode='auto', verbose=1, patience=3, factor=0.2, cooldown=1, min_delta=validation_min_delta)

    # Stops training entirely if, after 7 epochs, no improvement is found on the validation metric.
    # This triggers after the previous callback if it fails to make the training "great again".
    stopping_callback = EarlyStopping(patience=7, verbose=1, restore_best_weights=True, monitor=monitored_val_quantity, mode='auto', min_delta=validation_min_delta)

    save_callback = ModelCheckpoint(os.path.join(output_dir, model_name) + ".{epoch:02d}-{" + monitored_val_quantity + ":.4f}.hdf5", verbose=1, save_weights_only=False,
                                    save_best_only=True, monitor=monitored_val_quantity, mode='auto')

    history = model.fit_generator(training_data_generator, validation_data=validation_data_generator, initial_epoch=initial_epoch,
                                   epochs=training_epochs, callbacks=[save_callback, logger_callback, reduce_lr_plateau_callback, stopping_callback],
                                   use_multiprocessing=False)

    with open(os.path.join(output_dir, model_name + "_history"), 'wb') as f:
        print("Saving history ...")
        pickle.dump(history, f)

    model.save(os.path.join(output_dir, model_name + "_completed_training"))


def main():
    parser = argparse.ArgumentParser(description='Train model')
    parser.add_argument('-model', '--model_path', type=str, help='The path of the model', required=True)
    parser.add_argument('-metadata', '--metadata_path', type=str, help='The pathof the metadata file', required=True)
    parser.add_argument('-o', '--output_dir', type=str, help='The output dir', required=True)
    parser.add_argument('-ts', '--training_set_path', type=str, help='The path of the training set', required=True)
    parser.add_argument('-vs', '--validation_set_path', type=str, help='The path of the validation set', required=True)
    parser.add_argument('-e', '--epochs', type=int, help='Number of epochs', required=True)
    parser.add_argument('-ie', '--initial_epoch', type=int, help='Initial epoch', required=True)
    parser.add_argument('-v', '--verbose', action='store_true', help='Verbose')

    args = parser.parse_args()

    if args.verbose:
        print("Reading training and validation set")

    X_train, Y_train, f_train = read_dataset(args.training_set_path, args.verbose)
    X_val, Y_val, f_val = read_dataset(args.validation_set_path, args.verbose)

    train_model(args.model_path, args.metadata_path, args.output_dir, X_train, Y_train, X_val, Y_val,
                args.epochs, args.initial_epoch)

    f_train.close()
    f_val.close()


if __name__ == '__main__':
    main()