import tiny

data = [[0, 0, 0], [0, 1, 0], [1, 0, 0], [1, 1, 1]]  # AND

nn = tiny.network(activation='sigmoid')
nn.add_layer(2, 1)
nn.add_layer(1, 2)

nn.train(data, data, 0.5, 200)
