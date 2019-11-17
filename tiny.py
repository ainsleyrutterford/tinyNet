import numpy as np

class neuron:

    weights = []
    activation = 0
    delta = 0

    def __init__(self, inputs):
        self.weights = np.random.uniform(-1, 1, (inputs))
        activation = 0
        delta = 0


class network:

    neurons = []
    activation   = lambda: None
    activation_d = lambda: None

    def __init__(self, activation):
        self.neurons = []
        if activation == 'sigmoid':
            self.activation   = self.sigmoid
            self.activation_d = self.sigmoid_d

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_d(self, x):
        return x * (1 - x)

    def add_layer(self, inputs, outputs):
        self.neurons.append([neuron(inputs + 1) for i in range(outputs)])

    def forward_prop(self, inputs):
        for layer in self.neurons:
            layer_outputs = []
            for neuron in layer:
                bias = neuron.weights[-1]
                activation = self.activation(np.dot(neuron.weights[:-1], inputs) + bias)
                neuron.activation = activation
                layer_outputs.append(activation)
            inputs = layer_outputs
        return inputs

    def back_prop(self, expected):
        for i in reversed(range(len(self.neurons))):
            layer = self.neurons[i]
            errors = []
            if i == len(self.neurons)-1:
                for j, neuron in enumerate(layer):
                    errors.append(expected[j] - neuron.activation)
            else:
                for j in range(len(layer)):
                    error = 0.0
                    for neuron in self.neurons[i + 1]:
                        error += (neuron.weights[j] * neuron.delta)
                    errors.append(error)
            for j, neuron in enumerate(layer):
                neuron.delta = errors[j] * self.activation_d(neuron.activation)
        
    def update_weights(self, inputs, learning_rate):
        for i in range(len(self.neurons)):
            if i != 0:
                inputs = [neuron.activation for neuron in self.neurons[i - 1]]
            for neuron in self.neurons[i]:
                for j in range(len(inputs)):
                    neuron.weights[j] += learning_rate * neuron.delta * inputs[j]
                neuron.weights[-1] += learning_rate * neuron.delta

    def train(self, data, learning_rate, epochs):
        for epoch in range(epochs):
            sum_error = 0
            for sample in data:
                outputs = self.forward_prop(sample[:-1])
                label = sample[-1]
                expected = [0] * len(self.neurons[-1])
                expected[label] = 1
                sum_error += sum([(expected[i] - outputs[i])**2 for i in range(len(expected))])
                self.back_prop(expected)
                self.update_weights(sample[:-1], learning_rate)
            print(f'epoch {epoch}, error {sum_error}')