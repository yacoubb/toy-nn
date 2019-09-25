import numpy as np


class NeuralNetwork(object):
    def __init__(self, layers, activations_list):
        self.layers = layers
        self.weights = []
        self.biases = []

        for i in range(len(layers) - 1):
            self.weights.append(
                np.random.randn(layers[i + 1], layers[i])
            )  # note the weird w_jk notation here
            self.biases.append(np.random.randn(layers[i + 1], 1))