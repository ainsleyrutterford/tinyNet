import tiny

nn = tiny.network(activation='sigmoid')
nn.add_layer(3, 5)
nn.add_layer(5, 1)