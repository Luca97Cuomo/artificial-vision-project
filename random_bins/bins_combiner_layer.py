from keras.layers import Layer


class BinsCombinerLayer(Layer):
    def __init__(self, centroid_sets, **kwargs):
        super(BinsCombinerLayer, self).__init__(**kwargs)
        self.centroid_sets = centroid_sets

    def call(self, inputs):
        # todo in caso di lentezza o incompatibilità, adottare notazione funzionale sfruttando l'enum: https://stackoverflow.com/questions/50641219/equivalent-of-enumerate-in-tensorflow-to-use-index-in-tf-map-fn
        total_expected_value = 0
        for i, input in enumerate(inputs):
            total_expected_value += self._compute_single_bin_expected_value(input, i)
        return total_expected_value / len(inputs)

    def _compute_single_bin_expected_value(self, bin_output, bin_index):
        centroids = self.centroid_sets[bin_index]
        expected_value = 0
        for i, output in enumerate(bin_output):
            # ma comunque sembrano usare il centroide in https://github.com/axeber01/dold/blob/28f1386dcf44a7b6d42998009a4fbdf85af02849/age/scripts/utkRandomBins.m#L96
            expected_value += centroids[i] * output  # TODO ATTENZIONE IL CENTROIDE NON È IL MEAN VALUE DELL'INTERVALLO! COME LO OTTENGO?
        return expected_value

    def get_config(self):
        config = super(BinsCombinerLayer, self).get_config()
        config.update({"centroid_sets": self.centroid_sets})
        return config
