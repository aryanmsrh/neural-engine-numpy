class Optimizer:
    def step(self): raise NotImplementedError

class SGD(Optimizer): # stochastic gradient descent
    def __init__(self, layers, lr):
        self.layers = layers # trainable layers (excludes loss layer)
        self.lr = lr # learning rate hyperparameter

    def step(self):
        for layer in self.layers:
            if hasattr(layer, 'W'): # update layers with learnable parameters (W, B)
                layer.W -= self.lr * layer.dW # update weights
                layer.B -= self.lr * layer.dB # update biases

# optimizer module complete
