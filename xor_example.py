import numpy as np
from nn import NeuralNetwork
import matplotlib.pyplot as plt

np.random.seed(4)

network = NeuralNetwork([2,2,1])

x_train = [np.array([0, 0]), np.array([0, 1]), np.array([1, 0]), np.array([1, 1])]
x_train = list(map(lambda x: np.reshape(x, (2, 1)), x_train))
y_train = [np.array([0]), np.array([1]), np.array([1]), np.array([0])]
y_train = list(map(lambda x: np.reshape(x, (1, 1)), y_train))

print(x_train[0].shape)
print(y_train[0].shape)

for e in range(100):
    network.train(x_train, y_train, 100, 0.1)
    # print('=' * 20)
    # for elem in x_train:
    #     print(elem, network.predict(elem))

size = 32.0
img = np.zeros((int(size), int(size)))
for x in range(int(size)):
    for y in range(int(size)):
        coordinate = np.reshape(np.array([x / size, y / size]), (2, 1))
        img[x][y] = network.predict(coordinate) * 255


plt.imshow(img)
plt.show()
