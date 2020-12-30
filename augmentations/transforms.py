import skimage.color
import cv2
import numpy as np


def brightness_pure(x, c):
    x = np.array(x) / 255.
    if len(x.shape) > 2 and x.shape[2] > 1:
        x = skimage.color.rgb2hsv(x)
        x[:, :, 2] = np.clip(x[:, :, 2] + c, 0, 1)
        x = skimage.color.hsv2rgb(x)
    else:
        x = np.clip(x + c, 0, 1)

    return np.clip(x, 0, 1) * 255


def brightness_plus(x, severity=1):
    c = [.1, .2, .3, .4, .5][severity - 1]
    return brightness_pure(x, c)


def brightness_minus(x, severity=1):
    c = [.1, .2, .3, .4, .5][severity - 1]
    return brightness_pure(x, -c)


def contrast(x, severity):
    if severity < 0:
        severity *= -1
        c = [.4, .33, .24, .16, .1][severity - 1]
    elif severity > 0:
        c = [1.5, 1.9, 2.6, 3.3, 5.0][severity - 1]
    else:
        raise ValueError('Severity must be either positive or negative.')
    x = np.array(x) / 255.
    means = np.mean(x, axis=(0, 1), keepdims=True)
    return np.clip((x - means) * c + means, 0, 1) * 255


def brightness(x, severity):
    if severity < 0:
        severity *= -1
        return brightness_minus(x, severity)
    return brightness_plus(x, severity)


def gaussian_noise(x, severity, randomness):
    c = [.08, .12, 0.18, 0.24, 0.30][severity - 1]

    x = np.array(x) / 255.
    return np.clip(x + randomness.normal(size=x.shape, scale=c), 0, 1) * 255


def motion_blur(x, severity, vertical, horizontal):
    size = [5, 8, 10][severity - 1]
    kernel_motion_blur = np.zeros((size, size))
    if vertical:
        kernel_motion_blur[:, int((size - 1) / 2)] = np.ones(size)
    if horizontal:
        kernel_motion_blur[int((size - 1) / 2), :] = np.ones(size)
    kernel_motion_blur = kernel_motion_blur / (2*size if vertical and horizontal else size)
    return cv2.filter2D(x, -1, kernel_motion_blur)

