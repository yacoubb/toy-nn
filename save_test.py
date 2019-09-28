import numpy as np
from nn import NeuralNetwork

"""
A test of the effectiveness of the NeuralNetwork save and load functions.
"""

np.random.seed(4)
network = NeuralNetwork([5,5,2], 'Sigmoid')

network.save('./network.json')

loaded_network = NeuralNetwork.load('./network.json')

print(f"average error in first layer loaded weights: {np.mean(network.weights[0] - loaded_network.weights[0])}")
print(f"average error in first layer loaded biases: {np.mean(network.biases[0] - loaded_network.biases[0])}")

