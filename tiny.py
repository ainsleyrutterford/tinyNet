import numpy as np

def activate(weights, inputs):
        return np.dot(weights, inputs)

def sigmoid(activation):
    return 1 / (1 + np.exp(-activation))

def sigmoid_derivative():
    pass

class network:

    weights = []

    def add_layer(self, inputs, outputs):
        self.weights.append(np.random.rand(inputs, outputs))

    def train(self):
        pass

    def ponder(self, inputs):
        for layer in self.weights:
            layer_outputs = []
            for neuron_weights in layer.T:
                activation = activate(neuron_weights, inputs)
                layer_outputs.append(sigmoid(activation))
            inputs = layer_outputs
        return inputs