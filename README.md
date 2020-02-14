# tinyNet

[![Python version: 3.4+](https://img.shields.io/badge/python-3.4+-blue.svg)](https://www.python.org/download/releases/3.4.0/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

A tiny neural network framework implemented from scratch in Python. An example network is defined in `main.py` which is designed to compute logic gates. Another example network is defined in `mnist.py` which is designed to classify handwritten digits from the MNIST dataset.

## Example

### Creating a network

```python
import tiny

nn = tiny.network(activation='sigmoid')
nn.add_layer(784, 128)
nn.add_layer(128, 64)
nn.add_layer(64, 10)
```

### Training a network

```python
data       = ...
validation = ...

nn.train(data, validation, 0.01, 50)
```
