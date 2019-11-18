import numpy as np
from sklearn.metrics import log_loss

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
    error_history = []
    validation_accuracy_history = []
    validation_loss_history = []
    data_accuracy_history = []
    data_loss_history = []
    activation   = lambda: None
    activation_d = lambda: None

    def __init__(self, activation):
        self.neurons = []
        self.error_history = []
        self.validation_accuracy_history = []
        self.validation_loss_history = []
        self.data_accuracy_history = []
        self.data_loss_history = []
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

    def train(self, data, validation_data, learning_rate, epochs):
        for epoch in range(epochs):
            sum_error = 0
            validation_accuracy = 0
            validation_loss = 0
            data_accuracy = 0
            data_loss = 0
            for sample in validation_data:
                outputs = self.forward_prop(sample[:-1])
                label = sample[-1]
                expected = [0] * len(self.neurons[-1])
                expected[label] = 1
                if np.argmax(outputs) == label:
                    validation_accuracy += 1
                validation_loss += log_loss(expected, outputs)
            validation_accuracy /= len(validation_data)
            validation_loss /= len(validation_data)
            self.validation_accuracy_history.append(validation_accuracy)
            self.validation_loss_history.append(validation_loss)
            for sample in data:
                outputs = self.forward_prop(sample[:-1])
                label = sample[-1]
                if np.argmax(outputs) == label:
                    data_accuracy += 1
                expected = [0] * len(self.neurons[-1])
                expected[label] = 1
                data_loss += log_loss(expected, outputs)
                sum_error += sum([(expected[i] - outputs[i])**2 for i in range(len(expected))])
                self.back_prop(expected)
                self.update_weights(sample[:-1], learning_rate)
            data_accuracy /= len(data)
            data_loss /= len(data)
            self.data_accuracy_history.append(data_accuracy)
            self.data_loss_history.append(data_loss)
            self.error_history.append(sum_error)
            print(f'epoch {epoch}, error {sum_error} validation {validation_accuracy} accuracy {data_accuracy} val_loss {validation_loss} data_loss {data_loss}')