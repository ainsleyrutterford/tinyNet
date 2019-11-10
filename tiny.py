import numpy as np

class network:

    neurons = []
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
        self.neurons.append([0] * outputs)
        self.weights.append(np.random.rand(inputs, outputs))

    def forward_prop(self, inputs):
        for i, layer in enumerate(self.weights):
            layer_outputs = []
            for j, neuron_weights in enumerate(layer.T):
                activation = self.activation(np.dot(neuron_weights, inputs))
                self.neurons[i][j] = activation
                layer_outputs.append(activation)
            inputs = layer_outputs
        return inputs

    def train(self):
        pass