import numpy as np


class Layer():
    def forward(self, *args): raise NotImplementedError
    def backward(self, *args): raise NotImplementedError


class Dense(Layer):
    def __init__(self, input_dim, output_dim):
        self.W = np.random.randn(output_dim, input_dim) * np.sqrt(2.0 / input_dim) # he init
        self.B = np.zeros((output_dim, 1))

        # stored parameter gradients and cache for backprop
        self.dW = None
        self.dB = None
        self.A_prev = None # previous layer activations
        self.Z = None # current layer pre-activations

    def forward(self, A_prev):
        self.A_prev = A_prev
        self.Z = np.dot(self.W, self.A_prev) + self.B
        return self.Z

    def backward(self, dZ): # takes error term backpropagated from subsequent layer
        m = dZ.shape[1] # number of samples
        
        self.dW = (1/m) * np.dot(dZ, self.A_prev.T) # derived formula for dW
        self.dB = (1/m) * np.sum(dZ, axis=1, keepdims=True) # sum horizontally to get (n, 1) dims

        dA_prev = np.dot(self.W.T, dZ) # pass error upward to previous hidden layer (dA^[1] in derivation)

        return dA_prev
    

class ReLU(Layer):
    def __init__(self):
        self.Z = None
    
    def forward(self, Z):
        self.Z = Z
        return np.maximum(0, Z)

    def backward(self, dA):
        return dA * (self.Z > 0) # multiply incoming dA with relu derivative (hadamard product)
    

class SoftmaxCrossEntropy(Layer):
    def __init__(self):
        self.A = None
        self.Y = None

    def forward(self, Z, Y):
        self.Y = Y
        shift_Z = Z - np.max(Z, axis=0, keepdims=True) # subtract max z per column to prevent overflow
        # apply softmax
        self.A = np.exp(shift_Z) / (np.sum(np.exp(shift_Z), axis=0, keepdims=True)) # numpy broadcasting (nL, m) / (1, m)

        m = Y.shape[1] # number of samples
        eps = 1e-15 # clip probabilities to prevent log(0)
        self.A = np.clip(self.A, eps, 1.0 - eps)
        loss = -1/m * np.sum(Y * np.log(self.A)) # categorical cross entropy loss
        return loss
    
    def backward(self):
        return self.A - self.Y # derived gradient of fused softmax + cross-entropy loss
    
# layer module complete
# note: he init keeps variance close to 1. if variance > 1, weights blow up to nans; if < 1, weights collapse to zero (vanishing gradients). we use he init for relu since non-symmetric relu cuts variance in half.
