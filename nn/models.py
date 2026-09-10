import numpy as np


class Model:
    def forward(self, X): raise NotImplementedError
    def fit(self, X, Y, epochs, batch_size): raise NotImplementedError
    def predict(self, X): raise NotImplementedError


class Sequential(Model): # package everything together in sequence
    def __init__(self, layers, optimizer, learning_rate):
        self.layers = layers[:-1]
        self.loss_layer = layers[-1]
        self.optimizer = optimizer(self.layers, learning_rate) # instantiate passed optimizer
    
    def forward(self, X):
        out = X # input activations
        for layer in self.layers:
            out = layer.forward(out)
        return out # final hidden layer output
    
    def fit(self, X, Y, epochs, batch_size=64):
        # mini-batch vectorized training: in each epoch we update weights and biases
        # m/batch_size times, giving more frequent gradient updates than full-batch
        
        print("TRAINING\n")
        m = X.shape[1] # total number of training samples

        for epoch in range(epochs):
            perm = np.random.permutation(m) # random permutation of indices
            X_shuffled = X[:, perm] # shuffle data columns
            Y_shuffled = Y[:, perm] # shuffle labels in matching order

            epoch_loss = 0 # total loss in this epoch
            num_batches = 0 # number of batches processed

            for i in range(0, m, batch_size):
                X_batch = X_shuffled[:, i:i+batch_size]
                Y_batch = Y_shuffled[:, i:i+batch_size]

                out = self.forward(X_batch)
                epoch_loss += self.loss_layer.forward(out, Y_batch) # compute mini-batch loss
                num_batches += 1

                grad = self.loss_layer.backward() # start backprop from loss layer
                for layer in reversed(self.layers):
                    grad = layer.backward(grad) # backpropagate through hidden layers
                
                self.optimizer.step() # update weights and biases

            epoch_loss /= num_batches # average loss over mini-batches

            if epoch % 1 == 0 or epoch == epochs - 1: # evaluate full-dataset accuracy and loss
                out_full = self.forward(X) # forward pass without batching
                self.loss_layer.forward(out_full, Y)
                predictions = np.argmax(self.loss_layer.A, axis=0) # predicted class labels
                labels = np.argmax(Y, axis=0) # true class labels
                accuracy = np.mean(labels == predictions) * 100
                print(f"EPOCH: {epoch:3d} | LOSS: {epoch_loss:6f} | ACCURACY {accuracy:6f}%")

    def predict(self, X):
        out = self.forward(X)
        shift_out = out - np.max(out, axis=0, keepdims=True) # subtract max logit for numerical stability
        return np.exp(shift_out) / (np.sum(np.exp(shift_out), axis=0, keepdims=True)) # softmax probabilities via broadcasting

# model module complete
