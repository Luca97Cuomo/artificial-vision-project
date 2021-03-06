from keras.callbacks import ModelCheckpoint, EarlyStopping, CSVLogger, ReduceLROnPlateau
from keras import backend as K
import argparse
import pickle
from utils import *
import models
import preprocessing_functions
from preprocessing import load_labels
from generators import DataGenerator
from pathlib import Path
from augmentations import augmentation
import configuration
from utils import load_model


def train_model(configuration_file_path):
    conf = configuration.read_configuration(configuration_file_path)

    metadata_path = conf["metadata_path"]
    metadata = configuration.read_configuration(metadata_path)

    csv_path = conf["csv_path"]
    model_path = conf["model_path"]
    model_name = conf["model_name"]

    preprocessing = conf["preprocessing"]
    preprocessing_function_name = preprocessing["preprocessing_function_name"]
    enable_preprocessing = preprocessing["enabled"]

    tensorflow_version = conf["tf_version"]
    train = conf["train"]

    training_set_path = train["training_set_path"]
    num_training_samples = train["num_training_samples"]
    validation_set_path = train["validation_set_path"]
    num_validation_samples = train["num_validation_samples"]

    output_dir = train["output_training_dir"]

    learning_rate = train["train_learning_rate"]
    batch_size = train["batch_size"]
    initial_epoch = train["initial_epoch"]

    monitored_quantity = metadata["monitored_quantity"]
    normalization_function_name = metadata["normalization_function_name"]
    training_epochs = train["epochs"]

    input_shape = metadata["input_shape"]
    output_type = metadata["output_type"]

    augmentations = train["augmentations"]

    save_best_only = train["save_best_only"]

    verbose = conf["verbose"]

    print("Reading training and validation set")
    labels_dict = load_labels(csv_path, False)
    x_train, y_train = prepare_data_for_generator(training_set_path, labels_dict, num_training_samples)
    x_val, y_val = prepare_data_for_generator(validation_set_path, labels_dict, num_validation_samples)

    # create the checkpoints folder in the output folder
    checkpoints_dir = os.path.join(output_dir, "checkpoints")
    Path(checkpoints_dir).mkdir(parents=True, exist_ok=True)

    model = load_model(conf)

    if tensorflow_version == 2:
        if train.get("checkpoint_path") is not None:
            model.load_weights(train["checkpoint_path"])
        else:
            print("No checkpoint found, starting training from initial model!")

    if learning_rate is not None:
        if verbose:
            print("Changing the learning rate")
            print(f"The old learning rate is {K.eval(model.optimizer.lr)}")
        K.set_value(model.optimizer.learning_rate, learning_rate)
        if verbose:
            print(f"The new learning rate is {K.eval(model.optimizer.lr)}")

    normalization_function = models.NORMALIZATION_FUNCTIONS[normalization_function_name]

    n_outputs = 1

    if not enable_preprocessing:
        preprocessing_function = None
    elif preprocessing_function_name not in preprocessing_functions.AVAILABLE_PREPROCESSING_FUNCTIONS:
        raise Exception("The requested preprocessing function is not supported")
    else:
        preprocessing_function = preprocessing_functions.AVAILABLE_PREPROCESSING_FUNCTIONS[preprocessing_function_name]

    if output_type == "rvc":
        y_train = one_hot_encoded_labels(y_train, models.NUMBER_OF_RVC_CLASSES)
        y_val = one_hot_encoded_labels(y_val, models.NUMBER_OF_RVC_CLASSES)
    elif output_type == "random_bins_classification":
        y_train = models.BINNER.bin_labels(y_train)
        y_val = models.BINNER.bin_labels(y_val)
        n_outputs = models.BINNER.n_interval_sets

    if augmentations:
        random_seed = 42
        augmenter = augmentation.HorizontalMotionBlurAugmentation(probability=0.05, seed=random_seed)
        augmenter = augmentation.VerticalMotionBlurAugmentation(augmenter, probability=0.05, seed=random_seed)
        augmenter = augmentation.PixelateAugmentation(augmenter, probability=0.02, seed=random_seed)
        augmenter = augmentation.GaussianNoiseAugmentation(augmenter, probability=0.05, seed=random_seed)
        augmenter = augmentation.FlipAugmentation(augmenter, probability=0.2, seed=random_seed)
        augmenter = augmentation.BrightnessAugmentation(augmenter, probability=0.1, seed=random_seed)
        augmenter = augmentation.ContrastAugmentation(augmenter, probability=0.1, seed=random_seed)

    else:
        augmenter = None  # or augmentation.NullAugmentation()

    training_data_generator = DataGenerator(x_train, y_train, input_shape=input_shape, batch_size=batch_size,
                                            preprocessing_function=preprocessing_function,
                                            normalization_function=normalization_function, augmenter=augmenter,
                                            n_outputs=n_outputs)
    validation_data_generator = DataGenerator(x_val, y_val, input_shape=input_shape, batch_size=batch_size,
                                              preprocessing_function=preprocessing_function,
                                              normalization_function=normalization_function, n_outputs=n_outputs)

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
    reduce_lr_plateau_callback = ReduceLROnPlateau(monitor=monitored_quantity, mode='auto', verbose=1, patience=3,
                                                   factor=0.2, cooldown=1, min_delta=validation_min_delta)

    # Stops training entirely if, after 7 epochs, no improvement is found on the validation metric.
    # This triggers after the previous callback if it fails to make the training "great again".
    stopping_callback = EarlyStopping(patience=7, verbose=1, restore_best_weights=True, monitor=monitored_quantity,
                                      mode='auto', min_delta=validation_min_delta)

    save_callback = ModelCheckpoint(
        os.path.join(checkpoints_dir, model_name) + ".{epoch:02d}-{" + monitored_quantity + ":.4f}.hdf5", verbose=1,
        save_weights_only=False,
        save_best_only=save_best_only, monitor=monitored_quantity, mode='auto')

    history = model.fit(training_data_generator, validation_data=validation_data_generator,
                        initial_epoch=initial_epoch,
                        epochs=training_epochs,
                        callbacks=[save_callback, logger_callback, reduce_lr_plateau_callback, stopping_callback],
                        use_multiprocessing=False)

    with open(os.path.join(output_dir, model_name + "_history"), 'wb') as f:
        if verbose:
            print("Saving history ...")
        if tensorflow_version == 1:
            pickle.dump(history, f)
        else:
            pickle.dump(history.history, f)

    model.save(os.path.join(output_dir, model_name + "_completed_training"))


def main():
    parser = argparse.ArgumentParser(description='Train model')
    parser.add_argument('-c', '--configuration_file_path', type=str, help='The path of the configuration file',
                        required=True)

    args = parser.parse_args()

    train_model(args.configuration_file_path)


if __name__ == '__main__':
    main()
