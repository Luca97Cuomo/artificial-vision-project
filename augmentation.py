import typing
import numpy as np
from numpy.random import RandomState
from PIL import Image
from io import BytesIO
from utilities.corruptions import brightness_minus, brightness_plus, MotionImage


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


def motion_blur(x, severity, randomness):
    c = [(10, 3), (15, 5), (15, 8), (15, 12), (20, 15)][severity - 1]
    c = tuple([item / 48 * x.shape[0] for item in c])
    if len(x.shape) == 3 and x.shape[2] == 1:
        x = np.squeeze(x, 2)

    x = Image.fromarray(x[..., [2, 1, 0]].astype(np.uint8))

    output = BytesIO()
    x.save(output, format='PNG')
    x = MotionImage(blob=output.getvalue())

    x.motion_blur(radius=c[0] // 3, sigma=c[1] // 3,
                  angle=randomness.uniform(-45, 45))

    x = cv2.imdecode(np.frombuffer(x.make_blob(), np.uint8),
                     cv2.IMREAD_UNCHANGED)

    if len(x.shape) == 2:
        x = np.expand_dims(x, 2)
    return np.clip(x, 0, 255)


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
        self.randomness = RandomState(seed)

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


class GaussianNoiseAugmentation(SeverityAugmentation):
    def __init__(self, *args, **kwargs):
        if kwargs.get("severity_values") is None:
            kwargs["severity_values"] = [1, 2]

        super(GaussianNoiseAugmentation, self).__init__(*args, **kwargs)

    def _augmentation(self, image):
        return gaussian_noise(image, self.severity, self.randomness)


class MotionBlurAugmentation(SeverityAugmentation):
    def __init__(self, *args, **kwargs):
        if kwargs.get("severity_values") is None:
            kwargs["severity_values"] = [1]

        super(MotionBlurAugmentation, self).__init__(*args, **kwargs)

    def _augmentation(self, image):
        return motion_blur(image, self.severity, self.randomness)


if __name__ == '__main__':
    augmenter = MotionBlurAugmentation(probability=0.05)
    augmenter = GaussianNoiseAugmentation(augmenter, probability=0.05)
    augmenter = FlipAugmentation(augmenter, probability=0.2)
    augmenter = BrightnessAugmentation(augmenter, probability=0.2)
    augmenter = ContrastAugmentation(augmenter, probability=0.15)

    import cv2

    for i in range(20):
        lukino = cv2.imread("luchino_stanchino_forzutino.png")
        lukino = augmenter(lukino)
        cv2.imshow(r"test {i}", lukino)

        cv2.waitKey(0)
        cv2.destroyAllWindows()
