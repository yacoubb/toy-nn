import numpy as np
import sys
import os

sys.path.append(os.path.abspath(os.path.dirname(__file__)))
import losses
import activations


class NeuralNetwork(object):
    def __init__(self, layers, activations_list):
        # activations_list is a list of strings
        self.layers = layers
        assert len(layers) == len(activations_list) + 1, f"Layers and activations mismatch! {len(layers)} != {len(activations_list)} + 1"
        self.weights = []
        self.biases = []

        for i in range(len(layers) - 1):
            self.weights.append(np.random.randn(layers[i + 1], layers[i]))  # note the weird w_jk notation here
            self.biases.append(np.random.randn(layers[i + 1], 1))

        self.activations = list(map(lambda x: activations.str_to_activation(x), activations_list))
        self.cost = losses.MeanSquaredError

    def predict(self, z):
        a = np.copy(z)
        _, a_s = self.feedforward(z)
        return a_s[-1]

    def feedforward(self, z):
        a = np.copy(z)
        z_s = []
        a_s = [a]
        # note how the activations are shifted along one
        for i in range(len(self.weights)):
            z = np.dot(self.weights[i], a) + self.biases[i]
            z_s.append(z)
            a = self.activations[i].fn(np.copy(z))
            a_s.append(a)
        return (z_s, a_s)

    def backprop(self, x, y):
        (z_s, a_s) = self.feedforward(x)
        nabla_w_delta = []  # dC/dW
        nabla_b_delta = []  # dC/dB
        # dC/dW is the partial derivatives of the cost relative to all of the parameters in the network
        # so each element will be a matrix with the same dimensions as weight / bias matrix the element corresponds to
        deltas = [None for i in range(len(self.weights))]  # the errors at each layer delta^l

        # http://neuralnetworksanddeeplearning.com/chap2.html
        # BP1a:
        deltas[-1] = self.cost.derivative(y, a_s[-1]) * self.activations[-1].derivative(z_s[-1])

        # BP2:
        for i in reversed(range(len(deltas) - 1)):
            deltas[i] = self.weights[i + 1].T.dot(deltas[i + 1]) * self.activations[i].derivative(z_s[i])

        # BP3 and BP4:
        nabla_b_delta = [d.dot(np.ones((1, 1))) for d in deltas]
        # remember here that activation indices are shifted along one (look at feedforward method) so a_s[i] are activations in layer i-1
        nabla_w_delta = [d.dot(a_s[i].T) for i, d in enumerate(deltas)]

        return nabla_w_delta, nabla_b_delta

    def train(self, x_train, y_train, epochs, lr=0.01):
        for e in range(epochs):
            nabla_b = [0 for i in range(len(self.weights))]
            nabla_w = [0 for i in range(len(self.weights))]
            for i in range(len(y_train)):
                x = x_train[i]
                y = y_train[i]

                nabla_w_delta, nabla_b_delta = self.backprop(x, y)

                nabla_b = [b + db for b, db in zip(nabla_b, nabla_b_delta)]
                nabla_w = [w + dw for w, dw in zip(nabla_w, nabla_w_delta)]

            self.weights = [w + lr * nw for w, nw in zip(self.weights, nabla_w)]
            self.biases = [w + lr * nb for w, nb in zip(self.biases, nabla_b)]

    def save(self, path):
        with open(path, "w+") as save_file:
            import json

            save_obj = {"layers": self.layers, "activations": list(map(lambda x: x.to_string(), self.activations)), "weights": [], "biases": []}
            for l in range(len(self.weights)):
                save_obj["weights"].append(str(list(self.weights[l].flatten())))
                save_obj["biases"].append(str(list(self.biases[l].flatten())))

            save_file.write(json.dumps(save_obj, indent=4))

    @staticmethod
    def load(path):
        with open(path, "r") as network_file:
            import json

            json_object = json.load(network_file)
            layers = json_object["layers"]
            activations_list = json_object["activations"]

            network = NeuralNetwork(layers, activations_list)

            for l in range(len(network.weights)):
                loaded_weights = json_object["weights"][l][1:-1].split(", ")
                loaded_weights = list(map(lambda x: float(x), loaded_weights))
                loaded_weights = np.reshape(np.array(loaded_weights, np.float32), network.weights[l].shape)

                loaded_biases = json_object["biases"][l][1:-1].split(", ")
                loaded_biases = list(map(lambda x: float(x), loaded_biases))
                loaded_biases = np.reshape(np.array(loaded_biases, np.float32), network.biases[l].shape)

                network.weights[l] = loaded_weights
                network.biases[l] = loaded_biases

        return network

    def __repr__(self):
        return f"<nn {self.layers}, {sum(map(lambda x : len(x), self.weights)) + sum(map(lambda x : len(x), self.biases))} trainable params>"

    def copy(self):
        copy = NeuralNetwork(self.layers, list(map(lambda x: x.to_string(), self.activations)))
        for l in range(len(self.weights)):
            copy.weights[l] = np.copy(self.weights[l])
            copy.biases[l] = np.copy(self.biases[l])
        return copy
