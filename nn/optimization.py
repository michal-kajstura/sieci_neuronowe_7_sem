
def sgd(grads, layers, learning_rate):
    for layer, grad in zip(layers, grads):
        for weight in layer.keys():
            layer[weight] -= learning_rate * grad[weight]
