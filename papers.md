# Papers

## Deep Ordinal Regression with Label Diversity
- Download: https://arxiv.org/abs/2006.15864

### Notes
The RvC approach is used but multiple representations are used. The experimental results are similar to using the regression method directly.

## Age from Faces in the Deep Learning Revolution
- Download: https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8686239

### Notes
The best architecture seems to be Vgg16.
It seems that using an ensemble architecture improves the performance (like the RvC with multi representations).

#### Preprocessing
**Face detection**:
Deep learning algorithm techniques are more accurate but they are also more time consuming.
In the training phase we elaborate the images before the training, but in the inference phase we cannot do that. Considered that the speed is not relevant for this project, we can choose the most accurate detector.

**Pose normalization**:
It can be achieved using algorithms based on landmarks (the dlib library can be used) or using a face detector.
Regarding the face detector method, the image is rotated several times, for each time the detector is applied. It is choosen the rotaion that produce the highest accyuracy of the detector.
It seems that the algorithms based on landmarks achieves the best performances (but are computationally heavier).

**Rescaling**
Rescale the image to a given resolution.

**Intensity normalization**
Intensity nromalization is performed in order to reduce brightness change.

#### Data Augmentation
The most used technique is random cropping.
Ohter techniques are: traslation, scaling, flipping, rotation, sharpening, random brightness change, and addition noise.

#### Training
It is suggested to adopt the transfer learning approach. Choose a network pretrained (if possible on the age estimation task only), then substitute the dense layers and train only the dense layers with our dataset. It is important to choose a pretrained network trained for the most specific task correlated to age estimation. With a genral task the performance will decrease.

#### Architecture
It seems that the most effective architecture is Vgg16.

## A Data Augmentation Methodology to Improve Age Estimation using Convolutional Neural Networks
Download: https://ieeexplore.ieee.org/document/7813020

### Notes
It is proposed to perfom data augmentation using facial landmarks and algorithms that produce a trasformed image using this landmarks.
Doig so the, for example, it is possible for example to change the shape and the dimensions of some parts of the face, as the nose and the mouth.
