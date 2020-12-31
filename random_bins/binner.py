import typing
import numpy as np
from numpy.random import default_rng


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

    def bin_labels(self, labels: typing.Iterable[float]) -> typing.Iterable[typing.Iterable[float]]:
        new_labels = []
        for centroid_set in self.centroid_sets:
            new_labels.append([])
            for label in labels:
                label = round(label)
                interval_index = self._get_label_interval(label, centroid_set)
                new_label = np.zeros(self.n_intervals)
                new_label[interval_index] = 1
                new_labels[-1].append(new_label)

        return new_labels

    def architecture(self, output_layer):
        pass
