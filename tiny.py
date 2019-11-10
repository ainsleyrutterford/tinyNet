import numpy as np

class network:

    weights = []

    def add_layer(self, inputs, outputs):
        self.weights.append(np.random.rand(inputs, outputs))

    def activate(self, weights, inputs):
        return np.dot(weights, inputs)

    def sigmoid(self):
        pass

    def sigmoid_derivative(self):
        pass

    def train(self):
        pass

    def ponder(self):
        pass