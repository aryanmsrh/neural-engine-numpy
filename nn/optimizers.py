class Optimizer:
    def step(self, layers): raise NotImplementedError

class SGD(Optimizer):
    def __init__(self, lr=0.1):
        self.lr = lr

    def step(self, hidden_layers):
        for hidden_layer in hidden_layers:
            if hasattr(hidden_layer, 'W'):
                hidden_layer.W -= self.lr * hidden_layer.dW
                hidden_layer.B -= self.lr * hidden_layer.dB