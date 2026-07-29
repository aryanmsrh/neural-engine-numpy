import numpy as np


class Layer():
    def forward(self, *args): raise NotImplementedError
    def backward(self, *args): raise NotImplementedError


class Dense(Layer):
    def __init__(self, input_dim, output_dim):
        self.W = np.random.randn(output_dim, input_dim) * np.sqrt(2.0 / input_dim) # He Init.
        self.B = np.zeros((output_dim, 1))

        # Just storing it here, updates will be done by the Optimizer (SGD)
        self.dW = None
        self.dB = None
        self.A_prev = None # previous layer's activation
        self.Z = None # current layer's pre-act.

    def forward(self, A_prev):
        self.A_prev = A_prev
        self.Z = np.dot(self.W, self.A_prev) + self.B
        return self.Z

    def backward(self, dZ): # takes error term backpropagating/flowing up from previous layers
        m = dZ.shape[1]  # number of samples
        
        self.dW = (1/m) * np.dot(dZ, self.A_prev.T) # from our derived formula of dW.
        self.dB = (1/m) * np.sum(dZ, axis=1, keepdims=True) # summing horizontally. we end up with (n, 1) dims

        dA_prev = np.dot(self.W.T, dZ) # keeping track of this to pass this upward for the previous hidden layers [notice the dA^[1] we defined in our derivation]

        return dA_prev
    

class ReLU(Layer):
    def __init__(self):
        self.Z = None
    
    def forward(self, Z):
        self.Z = Z
        return np.maximum(0, Z)

    def backward(self, dA):
        return dA * (self.Z > 0) # multiply incoming dA (dA_prev from Dense) with derivative of ReLU layer. KEEP IN MIND THIS IS THE HAMMARD PRODUCT!
    

class SoftmaxCrossEntropy(Layer):
    def __init__(self):
        self.A = None
        self.Y = None

    def forward(self, Z, Y):
        self.Y = Y
        shift_Z = Z - np.max(Z, axis=0, keepdims=True) # find maximum Z vertically, subtract that from every element vertically in that columnn to prevent future overflow error
        # apply softmax
        self.A = np.exp(shift_Z) / (np.sum(np.exp(shift_Z), axis=0, keepdims=True)) # make use of numpy broadcasting to divide (nLxm) with (1xm)

        m = Y.shape[1] # number of samples
        loss = -1/m * np.sum(Y * np.log(self.A)) # categorical cross entropy loss
        return loss
    
    def backward(self):
        return self.A - self.Y # derived from combination of soft max act. + categorical cross entropy loss
    
# LAYER MODULE COMPLETE.
# note: we use He Init to have variance close to 1, if variance is more than 1, the weights might grow uncontrollably with training, hence the entire network becomes filled with NaNs, if the variance is < 1, the entire weights/bias/network would collapse to zero after many iterations. this is none as "vanishing gradients", we use He initalization instead of Xavier as we are using the ReLU activation unit which is not symmetrical, hence effectively killing half of the variance, unlike symmetric activation functions like sigmoid,tanh etc.
