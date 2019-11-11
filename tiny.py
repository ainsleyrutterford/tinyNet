import numpy as np

class neuron:
    activation = 0
    delta = 0

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
        self.neurons.append([neuron()] * outputs)
        self.weights.append(np.random.rand(inputs, outputs))

    def forward_prop(self, inputs):
        for i, layer in enumerate(self.weights):
            layer_outputs = []
            for j, neuron_weights in enumerate(layer.T):
                activation = self.activation(np.dot(neuron_weights, inputs))
                self.neurons[i][j].activation = activation
                layer_outputs.append(activation)
            inputs = layer_outputs
        return inputs
    
    def back_prop(self, expected):
        for i, layer_neurons in enumerate(reversed(self.neurons)):
            errors = []
            if (i == 0):
                for j, neuron in enumerate(layer_neurons):
                    errors.append(expected[j] - neuron.activation)
            else:
                for j in range(len(layer_neurons)):
                    error = 0.0
                    next_index = len(self.weights) - i
                    for k in range(len(self.weights[next_index][j])):
                        error += (self.weights[next_index][j][k] * self.neurons[next_index][k].delta)
                    errors.append(error) 
            for j, neuron in enumerate(layer_neurons):
                neuron.delta = errors[j] * self.activation_d(neuron.activation)

    def train(self):
        pass