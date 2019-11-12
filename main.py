import tiny

data = [[0, 0, 0, 0], # AND
        [0, 0, 1, 0],
        [0, 1, 0, 0],
        [0, 1, 1, 1],
        [1, 0, 0, 0], # OR 
        [1, 0, 1, 1],
        [1, 1, 0, 1],
        [1, 1, 1, 1],
        [2, 0, 0, 0], # XOR
        [2, 0, 1, 1],
        [2, 1, 0, 1],
        [2, 1, 1, 0]]

nn = tiny.network(activation='sigmoid')
nn.add_layer(3, 5)
nn.add_layer(5, 2)

nn.train(data, 0.5, 20)