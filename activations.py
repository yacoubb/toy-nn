import numpy as np

class Sigmoid(object):
    @staticmethod
    def fn(z):
        return 1.0 / (1.0 + np.exp(-z))

    @staticmethod
    def derivative(z):
        return Sigmoid.fn(z) * (1 - Sigmoid.fn(z))

    @staticmethod
    def to_string():
        return __class__.__name__

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

    @staticmethod
    def to_string():
        return __class__.__name__

class Linear(object):
    @staticmethod
    def fn(z):
        return z

    @staticmethod
    def derivative(z):
        return np.ones(z.shape)
    
    @staticmethod
    def to_string():
        return __class__.__name__


activations = [Sigmoid, Relu, Linear]
activation_dict = {}
for x in activations:
    activation_dict[x.to_string()] = x


def str_to_activation(label):
    assert label in activation_dict, f"{label} is not a valid activation!"
    return activation_dict[label]