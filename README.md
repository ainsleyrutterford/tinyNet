# tinyNet
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
