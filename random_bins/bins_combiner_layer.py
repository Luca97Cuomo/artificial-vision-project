from keras.layers import Layer
import tensorflow as tf
import numpy as np


class BinsCombinerLayer(Layer):
    def __init__(self, centroid_sets, **kwargs):
        super(BinsCombinerLayer, self).__init__(**kwargs)
        self.centroid_sets = centroid_sets

    def call(self, inputs):
        # https://stackoverflow.com/questions/50641219/equivalent-of-enumerate-in-tensorflow-to-use-index-in-tf-map-fn
        inputs_len = int(np.shape(inputs)[0])
        inputs_indices = np.arange(inputs_len)

        total_expected_value = tf.scan(
            lambda total, element: total + self._compute_single_bin_expected_value(element[0], element[1]),
            (inputs, inputs_indices)
        )

        return total_expected_value / inputs_len

    def _compute_single_bin_expected_value(self, bin_output, bin_index):
        # TODO ATTENZIONE IL CENTROIDE NON È IL MEAN VALUE DELL'INTERVALLO! COME LO OTTENGO?
        # ma comunque sembrano usare il centroide in https://github.com/axeber01/dold/blob/28f1386dcf44a7b6d42998009a4fbdf85af02849/age/scripts/utkRandomBins.m#L96
        centroids = self.centroid_sets[bin_index]

        bin_output_indices = np.arange(int(np.shape(bin_output)[0]))
        expected_value = tf.scan(lambda total, element: total + (element[0] * centroids[element[1]]),
                                 (bin_output, bin_output_indices))

        return expected_value

    def get_config(self):
        config = super(BinsCombinerLayer, self).get_config()
        config.update({"centroid_sets": self.centroid_sets})
        return config
