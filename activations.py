import numpy as np

class Sigmoid(object):
    @staticmethod
    def fn(z):
        return 1.0 / (1.0 + np.exp(-z))

    @staticmethod
    def derivative(z):
        return Sigmoid.fn(z) * (1 - Sigmoid.fn(z))


class Relu(object):
    @staticmethod
    def fn(z):
        z[z < 0] = 0
        return z

    @staticmethod
    def derivative(z):
        z[z >= 0] = 1
        z[z < 0] = 0
        return z


class Linear(object):
    @staticmethod
    def fn(z):
        return z

    @staticmethod
    def derivative(z):
        return np.ones(z.shape)
