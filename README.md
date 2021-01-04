# Age estimation framework

This framework allows you to build, train and test your net. You just need to modify a default configuration file.
This framework is not intended for production-grade use. It was created in order to speed up different experiments combining architectures and various parameters.

## Authors

```
Davide Cafaro
Carmine Carratu'
Luca Cuomo
Emanuele D'Arminio
```

## Installation

### Colab
It is strongly recommended to clone this repository on Google Colab instead cloning it on your personal machine.
This is because on Colab you don't have to install any dependencies and you won't have problem with CUDA libraries.

All you need to do is choose tensorflow version.

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
pip install git+https://github.com/rcmalli/keras-vggface.git
pip install -r requirements_tf1.txt

# TF 2
pip install -r requirements_tf2.txt
```
#### CUDA and cuDNN dependencies

Tested CUDA and cuDNN versions that works well with the framework:

```
- CUDA Toolkit: 10.1 (TF2)
- CUDA Toolkit: 10.0 (TF1)
- cuDNN: 7.6.5
```

## Usage

### Prepare the dataset

### Build your model

If you want to build your own model you need to write a json configuration file and pass it to the build model script.
You can see a template of the build configuration file:

```json
{
    "model_name": name for you model,
    "input_shape": input shape of your net, you can pass a json list like this [224, 224, 3],
    "output_type": choose one in ["regression", "rvc", "random_bins_classification"],
    "verbose": true or false,

    "build": {
      "build_model_dir": folder path where you want to save your model,
      "build_learning_rate": learning rate of your model (it must be a number),

      "dense_layer_structure_name": choose one in ["standard_dense_layer_structure", "vgg16_dense_layer_structure"],

      "backend": {
        "name": choose one in ["vgg16", "resnet50", "senet50"] if you're using TF1, choose one in ["vgg19"] if you're using TF2,
        "unlock_layers": this is the number of layers (from the end of the net) you want to unlock for training, choose one in ["none", "all", 1, 2, ..]
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

After created the configuration file just run the following script:

```shell script

python3 build_model.py -c "build_configuration_path"

```

The build model script will generate a metadata file, in your model output dir, useful for the training procedure. It will be something like this:

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

To train your net you need to write a json configuration file and pass it to the train model script.
You can see a template of the train configuration file:

```json

{
    "model_path": join(MODELS_PATH, MODEL_NAME, MODEL_FILE_NAME),
    "csv_path": join(DRIVE_BASE_PATH, "train.age_detected.csv"),
    "metadata_path": "",
    "model_name": MODEL_NAME,
    "tf_version": 2,
    "batch_size": 128,

    "verbose": True,

    "preprocessing": {
        "enabled": False,
        "preprocessing_function_name": "standard_preprocessing_function"
    },

    "train": {
        "training_set_path": "/content/training_set_resized",
        "validation_set_path": "/content/validation_set_resized",
        # if you want to train with tf 1 checkpoint_path must be specified in model_path
        "checkpoint_path": join(MODELS_PATH, MODEL_NAME, MODEL_CHECKPOINT),
        "num_training_samples": 300000,
        "num_validation_samples": 30000,
        "augmentations": True,
        "epochs": 20,
        "initial_epoch": 13,
        # train_learning_rate: if not None the learning rate of model will be changed
        "train_learning_rate": 0.00001,
        "output_training_dir": join(MODELS_PATH, MODEL_NAME),
        "save_best_only": False
    },
}

```

```shell script

!python3 train.py -c "train_configuration_path"

```

### Test your net