from keras.layers import Layer
import tensorflow as tf
import itertools
import numpy as np


class BinsCombinerLayer(Layer):
    def __init__(self, centroid_sets, **kwargs):
        super(BinsCombinerLayer, self).__init__(**kwargs)
        self.centroid_sets = centroid_sets
        # due to compat with tf1, i need to get this to be a tensor at some point.
        # since tensors cant be lists of lists of different sizes, i need to get the shape a square.
        # sets = len(self.centroid_sets)
        # set_dim = len(self.centroid_sets[0])

        # if sets > set_dim:
        #     new_centroid_sets = []
        #     for centroid_set in self.centroid_sets:
        #         new = np.zeros(sets)
        #         new[:set_dim] = centroid_set
        #         new_centroid_sets.append(new)
        #     self.centroid_sets = new_centroid_sets
        # elif set_dim > sets:
        #     for _ in range(set_dim - sets):
        #         self.centroid_sets.append(np.zeros(set_dim))

    def call(self, inputs):
        # https://stackoverflow.com/questions/50641219/equivalent-of-enumerate-in-tensorflow-to-use-index-in-tf-map-fn
        inputs = tf.reshape(inputs, (len(self.centroid_sets), len(self.centroid_sets[0])))
        inputs_len = tf.cast(tf.shape(inputs)[0], tf.float32)
        print(inputs_len)
        print(tf.shape(inputs))
        # inputs_indices = tf.range(inputs_len)

        self.i = 0  # hack
        accumulated_expected_values = tf.scan(
            lambda total, element: total + self._compute_single_bin_expected_value(element),
            inputs
        )
        total_expected_value = tf.gather_nd(accumulated_expected_values, (tf.shape(accumulated_expected_values)[0] - 1,))

        return total_expected_value / inputs_len

    def _compute_single_bin_expected_value(self, bin_output):
        # TODO ATTENZIONE IL CENTROIDE NON È IL MEAN VALUE DELL'INTERVALLO! COME LO OTTENGO?
        # ma comunque sembrano usare il centroide in https://github.com/axeber01/dold/blob/28f1386dcf44a7b6d42998009a4fbdf85af02849/age/scripts/utkRandomBins.m#L96
        centroids = tf.cast(tf.convert_to_tensor(self.centroid_sets[self.i]), tf.float32)

        # bin_output_indices = tf.range(tf.shape(bin_output)[0])
        # expected_value = tf.scan(lambda total, element: total + (tf.cast(element[0], tf.float32) * tf.cast(element[1], tf.float32)),
        #                          (bin_output, centroids))
        def scan(total, element):
            return element[0] * element[1] + total[0], 0.0
        expected_value = tf.scan(scan, (bin_output, centroids))[0]
        self.i += 1
        return expected_value

    def get_config(self):
        config = super(BinsCombinerLayer, self).get_config()
        config.update({"centroid_sets": self.centroid_sets})
        return config
