import random
import typing
import numpy as np
from utilities.corruptions import brightness_minus, brightness_plus


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


class AbstractAugmentation:
    def __init__(self,
                 augmenter: typing.Optional['AbstractAugmentation'] = None,
                 probability: float = 1.0,
                 seed: int = 42,
                 ):
        self.probability = probability
        # The last augmenter of the chain is always TypecastAugmentation in order to output np.uint8.
        # Most augmentations, in fact, typecast the input to float64.
        if augmenter is None and type(self) != TypecastAugmentation:
            augmenter = TypecastAugmentation()
        self.augmenter = augmenter
        self.randomness = random.Random(seed)

    def __call__(self, image):
        if self.randomness.random() < self.probability:
            image = self._augmentation(image)
        return self.augmenter(image) if self.augmenter else image

    def _augmentation(self, image):
        raise NotImplementedError('Must be implemented by subclasses')


class SeverityAugmentation(AbstractAugmentation):
    def __init__(self,
                 augmenter: typing.Optional['AbstractAugmentation'] = None,
                 probability: float = 1.0,
                 seed: int = 42,
                 severity_values=None):
        super().__init__(augmenter, probability, seed)
        if severity_values is None:
            severity_values = [-2, -1, 1, 2]
        self.severity_values = severity_values

    def _augmentation(self, image):
        raise NotImplementedError('Must be implemented by subclasses')

    @property
    def severity(self):
        return self.randomness.choice(self.severity_values)


class NullAugmentation(AbstractAugmentation):
    def _augmentation(self, image):
        return image


class TypecastAugmentation(AbstractAugmentation):
    def _augmentation(self, image):
        return image.astype(np.uint8)


class ContrastAugmentation(SeverityAugmentation):
    def _augmentation(self, image):
        return contrast(image, self.severity)


class BrightnessAugmentation(SeverityAugmentation):
    def _augmentation(self, image):
        return brightness(image, self.severity)


class FlipAugmentation(AbstractAugmentation):
    def _augmentation(self, image):
        return image[..., ::-1, :]

import cv2

# from utilities.corruptions import *
#
#
# def contrast_brightness_plus(x, severity):
#     sb, sc = [(1, 1), (2, 1), (2, 2), (2, 3), (3, 4)][severity - 1]
#     return contrast(brightness_plus(x, sb), sc)
#
#
# def contrast_brightness_minus(x, severity):
#     sb, sc = [(1, 1), (2, 1), (2, 2), (2, 3), (3, 4)][severity - 1]
#     return contrast(brightness_minus(x, sb), sc)
#
#
# def gaussian_noise_contrast_brightness_minus(x, severity):
#     sg, sb, sc = [(1, 1, 1), (2, 2, 1), (2, 2, 2), (3, 2, 3), (3, 2, 4)][severity - 1]
#     return contrast(brightness_minus(gaussian_noise(x, sg), sb), sc)
#
#
# def pixelate_contrast_brightness_minus(x, severity):
#     sp, sb, sc = [(1, 1, 1), (2, 2, 1), (3, 2, 2), (4, 2, 1), (4, 3, 3)][severity - 1]
#     return contrast(brightness_minus(pixelate(x, sp), sb), sc)
#
#
# def motion_blur_contrast_brightness_minus(x, severity):
#     sm, sb, sc = [(2, 1, 1), (3, 1, 1), (4, 2, 2), (5, 2, 1), (5, 2, 3)][severity - 1]
#     return contrast(brightness_minus(motion_blur(x, sm), sb), sc)
#
#
# orientation
augmentation = ContrastAugmentation(FlipAugmentation(BrightnessAugmentation()))

lukino = cv2.imread('luchino_stanchino_forzutino.png')

lukino = augmentation(lukino)
cv2.imshow('testlukino', lukino)
cv2.waitKey(0)
cv2.destroyAllWindows()

lukino = cv2.imread('luchino_stanchino_forzutino.png')

lukino = augmentation(lukino)
cv2.imshow('testlukino', lukino)
cv2.waitKey(0)
cv2.destroyAllWindows()
# resolution
# brightness

# class FlipAugmentation(AbstractAugmentation):
#   def __call__(self, image):
