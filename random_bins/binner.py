import typing
import numpy as np
from numpy.random import default_rng
from keras.layers import Dense, Concatenate


class Binner:
    def __init__(self, seed: int = 42, n_classes: int = 101, n_interval_sets: int = 30, n_intervals: int = 15):
        self.randomness = default_rng(seed=seed)

        self.n_classes = n_classes
        self.n_interval_sets = n_interval_sets
        self.n_intervals = n_intervals

        self._compute_centroids()

    def _compute_centroids(self):
        self.centroid_sets = []
        for i in range(self.n_interval_sets):
            random_centroids = self.randomness.choice(self.n_classes, size=self.n_intervals, replace=False)
            random_centroids.sort()
            self.centroid_sets.append(random_centroids)

    def _get_label_interval(self, label: int, centroid_set: np.ndarray) -> int:
        distance_from_centroids = abs(centroid_set - label)
        nearest_centroid_index = np.argmin(distance_from_centroids)
        return nearest_centroid_index

    def _one_hot_encode(self, class_index):
        new_label = np.zeros(self.n_intervals)
        new_label[class_index] = 1
        return new_label

    def bin_labels(self, labels: typing.Iterable[float]) -> typing.Iterable[typing.Iterable[float]]:
        new_labels = []
        for centroid_set in self.centroid_sets:
            new_labels.append([])
            for label in labels:
                interval_index = self._get_label_interval(round(label), centroid_set)
                new_labels[-1].append(self._one_hot_encode(interval_index))

        return new_labels

    def architecture(self, output_layer):
        outputs = []
        for _ in range(self.n_interval_sets):
            classifier = Dense(self.n_intervals, activation='softmax', kernel_initializer='glorot_normal')(output_layer)
            outputs.append(classifier)

        """
        return (outputs,
                ['categorical_crossentropy'] * self.n_interval_sets,
                [[]] * self.n_interval_sets,
                'val_loss')

        return outputs
        """

        return Concatenate(axis=0)(outputs)


    def compute_means(self):
        means_set = []
        for i in range(self.n_interval_sets):
            means = self._compute_classifier_means(self.centroid_sets[i])
            means_set.append(means)

        return means_set

    def _compute_classifier_means(self, centroids):
        means = []

        prec_centroid = None
        for i in range(len(centroids)):
            curr_centroid = centroids[i]
            if prec_centroid is None:
                left_border = 0
            else:
                left_border = (prec_centroid + curr_centroid) / 2

            if i == (len(centroids) - 1):
                right_border = self.n_classes - 1
            else:
                right_border = (curr_centroid + centroids[i + 1]) / 2

            mean = (left_border + right_border) / 2

            means.append(mean)

            prec_centroid = curr_centroid

        return means
