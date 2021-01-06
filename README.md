# Age estimation framework

This framework allows you to build, train and test your net. You just need to modify a default configuration file.
This framework is not intended for production-grade use. It was created in order to speed up different experiments combining architectures and various parameters.

## Authors

```
Davide Cafaro
Carmine Carratù
Luca Cuomo
Emanuele D'Arminio
```

## Installation

### Google Colab

It is strongly recommended to clone this repository on Google Colab instead of cloning it on your personal machine.
This is because on Colab you don't have to install any dependencies and you won't have problems with CUDA libraries.

All you need to do is to choose Tensorflow version.

```python 
%tensorflow_version 1 # 1 or 2

import tensorflow
print(tensorflow.__version__)
```

#### Tensorflow 1
If you want to use tensorflow 1 the only dependency to install is `keras-vggface`:

```shell script
pip install git+https://github.com/rcmalli/keras-vggface.git
```

### Personal computer

Clone this repository on your machine and execute:

```shell script
# TF 1
# Python versions supported 3.6 
pip install git+https://github.com/rcmalli/keras-vggface.git
pip install -r requirements_tf1.txt

# TF 2
# Python versions supported 3.6 - 3.8
pip install -r requirements_tf2.txt
```
#### CUDA and cuDNN dependencies

Tested CUDA and cuDNN versions that work well with the framework:

```
- CUDA Toolkit: 10.1 (TF2)
- CUDA Toolkit: 10.0 (TF1)
- cuDNN: 7.6.5
```

## Usage

### Prepare the dataset

If you want to preprocess your dataset in order to avoid doing it during the training (or the evaluation), saving a lot of computational time if executing multiple experiments with the same preprocessing, 
you can use the preprocessing module. We have assumed that your dataset is organized as follows:

```
dataset_path
|--- identity_1
|   |--- image_1.jpg
|   |--- image_2.jpg
|   |--- ...
|--- identity_2
|   |--- image_1.jpg
|   |--- image_2.jpg
|   |--- ...
|--- ...
```

Use the following script to divide the dataset in training-validation-test set and apply the preprocessing.

The preprocessing pipeline is:

- Detect the most relevant face into the image with OpenCV face detector.
- Crop the image with the enclosing square containing the face detected.
- Resize the cropped image with the input shape given as arguments.

```shell script

python3 preprocessing.py -d "dataset_path" -l "labels_path" -o "destination_path" -n "number_of_images_to_use" -v "validation_fraction" -t "test_fraction" -ih "new_image_height" -iw "new_image_width"

```

Using this script, the three datasets will contain different identities. In particular the `validation_fraction` and `test_fraction`, passed as arguments, refer to both identities and images of the split.  
We assumed that the labels are passed as a CSV file with this format:

```

identity_1/image_1.jpg,age_1
identity_1/image_2.jpg,age_2
...
identity_x/image_x.jpg,age_x
...

```

### Build your model

If you want to build your own model you need to write a JSON configuration file and pass it to the build model script.
You can see a template of the build configuration file:

```json
{
    "model_name": "name for your model",
    "input_shape": "input shape of your net, you can pass a json list like this [224, 224, 3]",
    "output_type": "choose one in ['regression', 'rvc', 'random_bins_classification']",
    "verbose": "boolean: true or false",

    "build": {
      "build_model_dir": "folder path where you want to save your model",
      "build_learning_rate": "learning rate of your model, it must be a number",

      "dense_layer_structure_name": "choose one in ['standard_dense_layer_structure', 'vgg16_dense_layer_structure']",

      "backend": {
        "name": "choose one in ['vgg16', 'resnet50', 'senet50'] if you're using TF1, choose one in ['vgg19'] if you're using TF2",
        "unlock_layers": "this is the number of layers (from the end of the net) you want to unlock for training, choose one in ['none', 'all', 1, 2, ..]"
      }
    }
}
```

For `"standard_dense_layer_structure"` we refer to:

```python

def standard_dense_layer_structure(backbone):
    global_pool = GlobalAveragePooling2D()
    x = global_pool(backbone)

    x = Dense(4096, activation='relu', kernel_initializer='he_normal')(x)
    x = Dense(2048, activation='relu', kernel_initializer='he_normal')(x)
    x = Dense(1024, activation='relu', kernel_initializer='he_normal')(x)

    return x

```

After having created the configuration file just run the following script to build your model:

```shell script

python3 build_model.py -c "build_configuration_path"

```

The build model script will generate a metadata file, in your model output directory, useful to scripts which use the model. It will be something like this:

```json
{
  "monitored_quantity": "val_mae",
  "normalization_function_name": "vgg16_normalization",
  "predict_function_name": "rvc_predict_function",
  "input_shape": [
    224,
    224,
    3
  ],
  "output_type": "rvc"
}
```

### Train your net

To train your net you need to write a JSON configuration file and pass it to the train model script.
You can see a template of the train configuration file:

```json

{
    "model_path": "path of the model, it is a directory if you're using TF2, it is a file if you're using TF1",
    "csv_path": "path of the labels saved in csv file",
    "metadata_path": "path of the metadata file generated by the build script",
    "model_name": "name of your model",
    "tf_version": "the version of tensorflow you're using now that must be equal to the version you used to build the model, choose one in [1, 2]",

    "verbose": "true or false",

    "preprocessing": {
        "enabled": "true or false, if true the training data will be preprocessed",
        "preprocessing_function_name": "choose one in ['standard_preprocessing_function']"
    },

    "train": {
        "training_set_path": "path of the training set",
        "validation_set_path": "path of the validation set",
        "checkpoint_path": "if you want to train with TF1, the checkpoint must be specified in model_path, this is useful only for models built with TF2.",
        "num_training_samples": "the number of training samples, it must be an integer",
        "num_validation_samples": "the number of validation samples, it must be an integer",
        "augmentations": "true or false, if true data will be randomly augmented with different corruptions during the training",
        "epochs": "number of epochs, it must be an integer",
        "batch_size": "the batch size you want to use, it must be an integer",
        "initial_epoch": "0 if you want to start from the first epoch, otherwise it can be set to resume training from a checkpoint correctly",
        "train_learning_rate": "if not null, the learning rate of the model chosen during the build will be overridden by this value",
        "output_training_dir": "directory path where checkpoints and model outputs will be stored",
        "save_best_only": "true or false, if false all the checkpoints are going to be saved, otherwise only the checkpoints which improve on the validation set"
    }
}

```

For `"standard_preprocessing_function"` we refer to the following preprocessing:

- Detect the most relevant face into the image with OpenCV face detector.
- Crop the image with the enclosing square containing the face detected.
- Resize the cropped image with the input shape given into the metadata file. 


After having created the configuration file just run the following script to train your model:
```shell script

python3 train.py -c "train_configuration_path"

```

### Test your net

To test your net you need to write a JSON configuration file and pass it to the evaluate model script.
You can see a template of the evaluate configuration file:

```json

{
    "model_path": "path of the model, it is a directory if you're using TF2, it is a file if you're using TF1",
    "csv_path": "path of the labels saved in csv file, null if you have to test your net on images without labels",
    "metadata_path": "path of the metadata file generated by the build script",
    "tf_version": "the tensorflow version that you're using now, choose one in [1, 2]",

    "verbose": "true or false",

    "preprocessing": {
        "enabled": "true or false, if true the test data will be preprocessed",
        "preprocessing_function_name": "choose one in ['standard_preprocessing_function']"
    },

    "evaluate": {
        "test_set_path": "absolute path of the test set",
        "num_test_samples": "number of test set samples you want to use, ignored if csv_path is null",

        "save_predictions": {
            "enabled": "true or false",
            "save_predictions_path": "path where to save predictions"
        },

        "age_intervals_evaluation": {
            "enabled": "true or false, if true the mae will be evaluated on different age intervals. Ignored if csv_path is null.",
            "age_interval_width": "number of age intervals to consider during age intervals evaluation, it must be a number. Ignored if csv_path is null."
        }
    }
}

```

The evaluation script can also be used as predict script if the path of the labels is null.

After having created the configuration file just run the following script to test your model:
```shell script

python3 evaluate.py -c "test_configuration_path"

```

### Demo

A demo has been developed to test your net. It works with images or with a camera.
Using the camera, you can also apply corruptions to the frames at runtime, to test how your net behaves with data corruption.
You can use the demo with Tensorflow 1 or 2, with or without GPU.

To execute the demo you need to create a configuration file as follow: 
```json

{
    "model_path": "path of the model saved as hdf5 file",
    "tf_version": "the tensorflow version that you're using now, choose one in [1, 2]",
    "input_shape": "input shape of your net, you can pass a json list like this [224, 224, 3]",
    "output_type": "choose one in ['regression', 'rvc', 'random_bins_classification']",    
    "verbose": "true or false",
    
    "build": {
        "backend": {
            "name": "choose one in ['vgg16', 'resnet50', 'senet50', 'vgg19']"
        }
    },

    "preprocessing": {
        "preprocessing_function_name": "choose one in ['demo_preprocessing']"
    },

    "evaluate": {
        "test_set_path": "absolute path of folder containing the images to use for the demo. Ignored if you want to use the camera.",
        "save_predictions": {
            "enabled": "true or false",
            "save_predictions_path": "path where to save predictions"
        }
    }
}

```

After created the configuration file just run the following script to test your model with the demo:

```shell script

# Run with camera
python3 age_estimator.py -c "demo_configuration_path"

# Run on images
python3 age_estimator.py -c "demo_configuration_path" -i

```
The corruption that you can apply to the frames captured by your camera are additive and have different severity values.

#### How to use the demo with camera

```
How to use the demo:

Press ESC to quit.

Press SPACE to save prediction in <save_prediction_path>.

- Press 1 for horizontal motion blur

- Press 2 for vertical motion blur

- Press 3 for pixelate motion blur

- Press 4 for gaussian noise corruption

- Press 5 for brightness change corruption

- Press 6 for contrast change corruption

- Press R to reset all the corruptions

Every time you press a number the severity value is increased, if it reaches the severity limit it will be deactivated.

```

The instructions will be printed also after you execute the script.



