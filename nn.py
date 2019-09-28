import numpy as np


class Sigmoid(object):
    @staticmethod
    def fn(z):
        return 1.0 / (1.0 + np.exp(-z))

    @staticmethod
    def derivative(z):
        return Sigmoid.fn(z) * (1 - Sigmoid.fn(z))

class MeanSquaredError(object):
    @staticmethod
    def fn(y, a):
        return 0.5 * (y - a) ** 2

    @staticmethod
    def derivative(y, a):
        return y - a

class NeuralNetwork(object):
    def __init__(self, layers):
        self.layers = layers
        self.weights = []
        self.biases = []

        for i in range(len(layers) - 1):
            self.weights.append(
                np.random.randn(layers[i + 1], layers[i])
            )  # note the weird w_jk notation here
            self.biases.append(np.random.randn(layers[i + 1], 1))

        self.activation = Sigmoid
        self.cost = MeanSquaredError


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
            a = self.activation.fn(np.copy(z))
            a_s.append(a)
        return (z_s, a_s)

    def backprop(self, x, y):
        (z_s, a_s) = self.feedforward(x)
        nabla_w_delta = []  # dC/dW
        nabla_b_delta = []  # dC/dB
        # dC/dW is the partial derivatives of the cost relative to all of the parameters in the network
        # so each element will be a matrix with the same dimensions as weight / bias matrix the element corresponds to
        deltas = [
            None for i in range(len(self.weights))
        ]  # the errors at each layer delta^l

        # http://neuralnetworksanddeeplearning.com/chap2.html
        # BP1a:
        deltas[-1] = self.cost.derivative(y, a_s[-1]) * self.activation.derivative(
            z_s[-1]
        )

        # BP2:
        for i in reversed(range(len(deltas) - 1)):
            deltas[i] = self.weights[i + 1].T.dot(deltas[i + 1]) * self.activation.derivative(z_s[i])

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