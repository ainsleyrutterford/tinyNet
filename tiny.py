import numpy as np

class neuron:

    weights = []
    activation = 0
    delta = 0

    def __init__(self, inputs):
        self.weights = np.random.uniform(-1, 1, (inputs))
        # if (inputs == 1):
        #     self.weights = np.array([0.13436424411240122, 0.8474337369372327, 0.763774618976614])
        # elif (inputs == 2):
        #     self.weights = np.array([0.2550690257394217, 0.49543508709194095])
        # elif (inputs == 3):
        #     self.weights = np.array([0.4494910647887381, 0.651592972722763])

class network:

    neurons = []
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
        self.neurons.append([neuron(inputs + 1) for i in range(outputs)])
        # if (outputs == 1):
        #     self.neurons.append([neuron(1)])
        # elif (outputs == 2):
        #     self.neurons.append([neuron(2),neuron(3)])

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

    def backward_prop(self, expected):
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
