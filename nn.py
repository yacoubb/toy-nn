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

    def activation(self, z):
        return 1.0 / (1.0 + np.exp(-z))


    def predict(self, z):
        _, a_s = self.feedforward(z)
        return a_s[-1]

    def feedforward(self, x):
        a = np.copy(x)
        z_s = []
        a_s = [a]
        for i in range(len(self.weights)):
            z = np.dot(self.weights[i], a) + self.biases[i]
            z_s.append(z)
            a = self.activation(np.copy(z))
            a_s.append(a)
        return (z_s, a_s)

    