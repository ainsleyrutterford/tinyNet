import numpy as np

class network:

    weights = []
    activation   = lambda: None
    activation_d = lambda: None

    def __init__(self, activation):
        if activation == 'sigmoid':
            self.activation   = self.sigmoid
            self.activation_d = self.sigmoid_d

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_d(self, x):
        return x * (1 - x)

    def add_layer(self, inputs, outputs):
        self.weights.append(np.random.rand(inputs, outputs))

    def train(self):
        pass

    def ponder(self, inputs):
        for layer in self.weights:
            layer_outputs = []
            for neuron_weights in layer.T:
                weighted_sum = np.dot(neuron_weights, inputs)
                layer_outputs.append(self.activation(weighted_sum))
            inputs = layer_outputs
        return inputs