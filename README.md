# Age estimation framework

This framework allows you to build, train and test your net. You just need to modify a default configuration file.


## Installation

### Colab
It is strongly recommended to clone this repository on Google Colab instead cloning it on your personal machine.
This is because on Colab you don't have to install any dependencies and you won't have problem with CUDA libraries.

All you need to do is choose tensorflow version.

```python 
%tensorflow_version 1

import tensorflow
print(tensorflow.__version__)
```

#### Tensorflow 1
If you want to use tensorflow 1 there are three available backends for your net: _vgg16, resnet50, senet50_.

The only dependency to install is `keras-vggface`:

```shell script
pip install git+https://github.com/rcmalli/keras-vggface.git
```

#### Tensorflow 2
Otherwise use tensorflow 2 if you want _vgg19_ as backend for your net. 

### Personal computer

Clone this repository on your machine and execute:

```shell script
# TF 1
pip install git+https://github.com/rcmalli/keras-vggface.git
pip install -r requirements_tf1.txt

# TF 2
pip install -r requirements_tf2.txt
```

#### 