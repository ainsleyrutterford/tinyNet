import numpy as np

class network:

    weights = []
    activation   = lambda: None
    activation_d = lambda: None

    def __init__(self, activation):
        if activation == 'sigmoid':
            self.activation   = self.sigmoid
            self.activation_d = self.sigmoid_d

    def sigmoid(self, weights, inputs):
        return 1 / (1 + np.exp( -np.dot(weights, inputs) ))

    def sigmoid_d(self):
        pass

    def add_layer(self, inputs, outputs):
        self.weights.append(np.random.rand(inputs, outputs))

    def train(self):
        pass

    def ponder(self, inputs):
        for layer in self.weights:
            layer_outputs = []
            for neuron_weights in layer.T:
                layer_outputs.append(self.activation(neuron_weights, inputs))
            inputs = layer_outputs
        return inputs