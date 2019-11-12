import tiny

data = [[0, 0, 0], # AND
        [0, 1, 0],
        [1, 0, 0],
        [1, 1, 1]]

nn = tiny.network(activation='sigmoid')
nn.add_layer(2, 4)
nn.add_layer(4, 1)

nn.train(data, 1, 20000)

print(nn.forward_prop([0,0]))
print(nn.forward_prop([0,1]))
print(nn.forward_prop([1,0]))
print(nn.forward_prop([1,1]))