import typing
import numpy as np
from numpy.random import RandomState
import transforms


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
            # print(f"before augmentation image shape {image.shape}. Type: {type(self)}")
            image = self._augmentation(image)
            # print(f"after augmentation image shape {image.shape}. Type: {type(self)}")
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
        return transforms.contrast(image, self.severity)


class BrightnessAugmentation(SeverityAugmentation):
    def _augmentation(self, image):
        return transforms.brightness(image, self.severity)


class FlipAugmentation(AbstractAugmentation):
    def _augmentation(self, image):
        return image[..., ::-1, :]


class GaussianNoiseAugmentation(SeverityAugmentation):
    def __init__(self, *args, **kwargs):
        if kwargs.get("severity_values") is None:
            kwargs["severity_values"] = [1, 2]

        super(GaussianNoiseAugmentation, self).__init__(*args, **kwargs)

    def _augmentation(self, image):
        return transforms.gaussian_noise(image, self.severity, self.randomness)


class HorizontalMotionBlurAugmentation(SeverityAugmentation):
    def __init__(self, *args, **kwargs):
        if kwargs.get("severity_values") is None:
            kwargs["severity_values"] = [1, 2, 3]

        super(HorizontalMotionBlurAugmentation, self).__init__(*args, **kwargs)

    def _augmentation(self, image):
        return transforms.motion_blur(image, self.severity, False, True)


class VerticalMotionBlurAugmentation(SeverityAugmentation):
    def __init__(self, *args, **kwargs):
        if kwargs.get("severity_values") is None:
            kwargs["severity_values"] = [1, 2, 3]

        super(VerticalMotionBlurAugmentation, self).__init__(*args, **kwargs)

    def _augmentation(self, image):
        return transforms.motion_blur(image, self.severity, True, False)


class PixelateAugmentation(SeverityAugmentation):
    def __init__(self, *args, **kwargs):
        if kwargs.get("severity_values") is None:
            kwargs["severity_values"] = [1, 2, 3]

        super(PixelateAugmentation, self).__init__(*args, **kwargs)

    def _augmentation(self, image):
        return transforms.motion_blur(image, self.severity, True, True)


if __name__ == '__main__':
    import cv2
    import pathlib
    augmenter = HorizontalMotionBlurAugmentation(probability=0.05)
    augmenter = VerticalMotionBlurAugmentation(augmenter, probability=0.05)
    augmenter = PixelateAugmentation(augmenter, probability=0.02)
    augmenter = GaussianNoiseAugmentation(augmenter, probability=0.05)
    augmenter = FlipAugmentation(augmenter, probability=0.2)
    augmenter = BrightnessAugmentation(augmenter, probability=0.1)
    augmenter = ContrastAugmentation(augmenter, probability=0.1)

    path = pathlib.Path(".").parent / 'utilities' / 'luchino_stanchino_forzutino.png'
    path = str(path.absolute())
    lukino = cv2.imread(path)
    lukino = cv2.resize(lukino, (224, 224))
    cv2.imshow("test1", lukino)
    lukino = augmenter(lukino)
    cv2.imshow("test2", lukino)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
