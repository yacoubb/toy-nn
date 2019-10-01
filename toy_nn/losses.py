import numpy as np


class MeanSquaredError(object):
    @staticmethod
    def fn(y, a):
        return 0.5 * (y - a) ** 2

    @staticmethod
    def derivative(y, a):
        return y - a
