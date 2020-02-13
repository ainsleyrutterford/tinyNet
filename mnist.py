import idx2numpy
import numpy as np
import pickle
import tiny

np.seterr(over='ignore')

samples = 10000

training_images = idx2numpy.convert_from_file('mnist/train-images-idx3-ubyte')[:samples]
training_labels = idx2numpy.convert_from_file('mnist/train-labels-idx1-ubyte')[:samples]
training_images = training_images.reshape(samples, 784)

training_data = np.c_[training_images, training_labels]

test_images = idx2numpy.convert_from_file('mnist/t10k-images-idx3-ubyte')
test_labels = idx2numpy.convert_from_file('mnist/t10k-labels-idx1-ubyte')
test_images = test_images.reshape(len(test_images), 784)

test_data = np.c_[test_images, test_labels]

nn = tiny.network(activation='sigmoid')
nn.add_layer(784, 128)
nn.add_layer(128, 64)
nn.add_layer(64, 10)

# f = open('saved_mnist_tiny', 'rb')
# nn = pickle.load(f)
# f.close()

nn.train(training_data, test_data, 0.02, 20)

file_name = 'saved_mnist_tiny'
print(f'Saving network in: {file_name}')
f = open(file_name, 'wb')
pickle.dump(nn, f)
f.close()
