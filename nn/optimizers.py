class Optimizer:
    def step(self): raise NotImplementedError

class SGD(Optimizer): # stochastic gradient descent
    def __init__(self, layers, lr):
        self.layers = layers # the layers to update (excludes the loss layer)
        self.lr = lr # the hyperparameter

    def step(self):
        for layer in self.layers:
            if hasattr(layer, 'W'): # only the dense layers have W,B, so we cannot update every layer
                layer.W -= self.lr * layer.dW # update layer weight with stored delta/error
                layer.B -= self.lr * layer.dB # update layer bias with stored delta/error

# OPTIMIZER MODULE COMPLETE.
