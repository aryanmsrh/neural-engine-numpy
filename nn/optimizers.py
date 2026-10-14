class Optimizer:
    def step(self, layers): raise NotImplementedError


class SGD(Optimizer):
    def __init__(self, lr=0.1):
        self.lr = lr

    def __repr__(self):
        return f"SGD(lr={self.lr})"

    def step(self, hidden_layers):
        for hidden_layer in hidden_layers:
            if hasattr(hidden_layer, 'W'):
                hidden_layer.W -= self.lr * hidden_layer.dW
                hidden_layer.B -= self.lr * hidden_layer.dB


class RMSProp(Optimizer):
    def __init__(self, lr=0.1, beta_2=0.99):
        self.lr = lr
        self.beta_2 = beta_2

    def __repr__(self):
        return f"RMSProp(lr={self.lr}, beta_2={self.beta_2})"

    def step(self, hidden_layers):
        pass