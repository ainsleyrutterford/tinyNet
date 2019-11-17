import idx2numpy
import numpy as np
import pickle
import tiny

np.seterr(over='ignore')

start_sample = 0
end_sample = 50

training_images = idx2numpy.convert_from_file('mnist/train-images-idx3-ubyte')[start_sample:end_sample]
training_labels = idx2numpy.convert_from_file('mnist/train-labels-idx1-ubyte')[start_sample:end_sample]
training_images = training_images.reshape(end_sample - start_sample, 784)

training_data = np.c_[ training_images, training_labels ]

nn = tiny.network(activation='sigmoid')
nn.add_layer(784, 200)
nn.add_layer(200, 10)

nn.train(training_data, 0.7, 5)

file_name = 'saved_mnist_tiny'
print(f'Saving network in: {file_name}')
f = open(file_name, 'wb')
pickle.dump(nn, f)
f.close()

# f2 = open('saved_mnist_tiny', 'rb')
# new_nn = pickle.load(f2)
# f2.close()