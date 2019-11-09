import tiny

nn = tiny.network()
nn.add_layer(3, 5)
nn.add_layer(5, 1)

print(nn.weights)