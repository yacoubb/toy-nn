import numpy as np
import os, sys

sys.path.insert(0, os.path.join(os.path.abspath(os.path.dirname(__file__)), ".."))
from toy_nn.nn import NeuralNetwork


def test_copy():
    """
    A test of the NeuralNetwork copy function.
    """

    np.random.seed(4)
    network = NeuralNetwork([5, 5, 2], ["Sigmoid", "Sigmoid"])

    copy = network.copy()
    assert len(network.layers) == len(copy.layers)
    assert len(network.activations) == len(copy.activations)
    for l in range(len(network.weights)):
        assert np.array_equal(network.weights[l], copy.weights[l])
        assert np.array_equal(network.biases[l], copy.biases[l])


def test_save():
    """
    A test of the effectiveness of the NeuralNetwork save and load functions.
    """

    np.random.seed(4)
    network = NeuralNetwork([5, 5, 2], ["Sigmoid", "Sigmoid"])
    assert not os.path.exists("./network.json"), "test would overwrite network.json!"
    network.save("./network.json")

    loaded_network = NeuralNetwork.load("./network.json")

    # since the weights and biases are saved in text form, some precision is lost.
    # need to do further testing to find out if this is going to be an issue
    print(f"average error in first layer loaded weights: {np.mean(network.weights[0] - loaded_network.weights[0])}")
    print(f"average error in first layer loaded biases: {np.mean(network.biases[0] - loaded_network.biases[0])}")

    os.remove("./network.json")
