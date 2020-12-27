from keras.callbacks import ModelCheckpoint, EarlyStopping, CSVLogger, ReduceLROnPlateau
import pickle
import keras
import os
import json
import argparse
from utils import read_dataset
import models
from generators import TrainDataGenerator
import h5py


def train_model(model_path, metafile_path, output_dir, batch_size, training_path, validation_path, training_epochs, initial_epoch):

    model_name = os.path.basename(model_path)
    model = keras.models.load_model(model_path)

    metadata = None
    with open(metafile_path) as json_file:
        metadata = json.load(json_file)

    monitored_val_quantity = metadata["val_metric_name"]
    normalization_function_name = metadata["normalization_function_name"]
    normalization_function = models.NORMALIZATION_FUNCTIONS[normalization_function_name]

    with h5py.File(training_path, "r") as f_train:
        num_training_samples = len(f_train["X"])

    with h5py.File(validation_path, "r") as f_val:
        num_validation_samples = len(f_val["X"])

    training_data_generator = TrainDataGenerator(training_path, num_training_samples, batch_size=batch_size, normalization_function=normalization_function)
    validation_data_generator = TrainDataGenerator(validation_path, num_validation_samples, batch_size=batch_size, normalization_function=normalization_function)

    append = None
    if initial_epoch == 0:
        append = False
    else:
        append = True
    logger_callback = CSVLogger(os.path.join(output_dir, model_name) + ".log", append=append)

    # divides learning rate by 5 if no improvement is found on validation over 3 epochs.
    # this is made in order to avoid stagnating too much: empirical evidence suggests that reducing learning rate when stagnating
    # can significantly help a model training process. This is probably because it can stagnate because of strange shapes
    # in the loss function which make it impossible for the optimizer to fall in better minima.
    validation_min_delta = 0.1 / 100  # minimum variation of the monitored quantity for it to be considered improved
    reduce_lr_plateau_callback = ReduceLROnPlateau(monitor=monitored_val_quantity, mode='auto', verbose=1, patience=3,
                                                   factor=0.2, cooldown=1, min_delta=validation_min_delta)

    # Stops training entirely if, after 7 epochs, no improvement is found on the validation metric.
    # This triggers after the previous callback if it fails to make the training "great again".
    stopping_callback = EarlyStopping(patience=7, verbose=1, restore_best_weights=True, monitor=monitored_val_quantity,
                                      mode='auto', min_delta=validation_min_delta)

    save_callback = ModelCheckpoint(
        os.path.join(output_dir, model_name) + ".{epoch:02d}-{" + monitored_val_quantity + ":.4f}.hdf5", verbose=1,
        save_weights_only=False,
        save_best_only=True, monitor=monitored_val_quantity, mode='auto')

    history = model.fit(training_data_generator, validation_data=validation_data_generator,
                        initial_epoch=initial_epoch,
                        epochs=training_epochs,
                        callbacks=[save_callback, logger_callback, reduce_lr_plateau_callback, stopping_callback],
                        use_multiprocessing=True, workers=10)

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
    parser.add_argument('-b', '--batch_size', type=int, help='batch size', required=True)

    args = parser.parse_args()

    if args.verbose:
        print("Reading training and validation set")


    train_model(args.model_path, args.metadata_path, args.output_dir, args.batch_size, args.training_set_path, args.validation_set_path,
                args.epochs, args.initial_epoch)

    f_train.close()
    f_val.close()

    r"""
        python -model "C:\Users\utente\Desktop\Magistrale\magistrale\Visione artificiale\test\modelli_di_test\resnet50_regression\resnet50_regression_model" 
        -metadata "C:\Users\utente\Desktop\Magistrale\magistrale\Visione artificiale\test\modelli_di_test\resnet50_regression\resnet50_regression_metadata.txt"
        -o "C:\Users\utente\Desktop\Magistrale\magistrale\Visione artificiale\test\risultato_training_di_test" 
        -ts "C:\Users\utente\Desktop\Magistrale\magistrale\Visione artificiale\test\h5_di_test\training_set.h5"
        -vs "C:\Users\utente\Desktop\Magistrale\magistrale\Visione artificiale\test\h5_di_test\validation_set.h5"
        -e 10 -ie 0 -v -b 64
    """


if __name__ == '__main__':
    main()
