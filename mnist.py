import idx2numpy
import numpy as np
import tiny

training_images = idx2numpy.convert_from_file('mnist/train-images-idx3-ubyte')[:10000]
training_labels = idx2numpy.convert_from_file('mnist/train-labels-idx1-ubyte')[:10000]
training_images = training_images.reshape(10000,784)

training_data = np.c_[ training_images, training_labels ]