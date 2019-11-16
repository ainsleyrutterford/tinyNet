import idx2numpy
import numpy as np
import tiny

np.seterr(over='ignore')

samples = 1000

training_images = idx2numpy.convert_from_file('mnist/train-images-idx3-ubyte')[:samples]
training_labels = idx2numpy.convert_from_file('mnist/train-labels-idx1-ubyte')[:samples]
training_images = training_images.reshape(samples,784)

training_data = np.c_[ training_images, training_labels ]

nn = tiny.network(activation='sigmoid')
nn.add_layer(784, 200)
nn.add_layer(200, 10)

nn.train(training_data, 0.7, 1000)