import numpy as np


class Model:
    def forward(self, X): raise NotImplementedError
    def fit(self, X, Y, epochs, batch_size): raise NotImplementedError
    def predict(self, X): raise NotImplementedError


class Sequential(Model): # package everything togeher in a sequence
    def __init__(self, layers, optimizer, learning_rate):
        self.layers = layers[:-1]
        self.loss_layer = layers[-1]
        self.optimizer = optimizer(self.layers, learning_rate) # init optimizer class that is passed
    
    def forward(self, X):
        out = X # take current/latest activations
        for layer in self.layers:
            out = layer.forward(out)
        return out # final output of last/loss layer
    
    def fit(self, X, Y, epochs, batch_size=64):
        # implementing MINI-BATCH VECTORIZED TRAINING instead of full-batch training
        # through MINI-BATCH, in one single epoch, we are updating the weights and biases
        # N/b amount of times, which is `b` times more than the normal number of updates. N is sample size
        # which we would have done in a full-batch training. more updates generally train a better network
        
        print("TRAINING\n")
        m = X.shape[1] # the number of elements in the entire training sample

        for epoch in range(epochs):
            perm = np.random.permutation(m) # array of size m, containing random numbers <m
            X_shuffled = X[:, perm] # shuffle the entire data by slicing and putting back together
            Y_shuffled = Y[:, perm] # shuffle the labels too, but in the same ORDER!!

            epoch_loss = 0 # total loss in this epoch
            num_batches = 0 # number of batches completed

            for i in range(0, m, batch_size):
                X_batch = X_shuffled[:, i:i+batch_size]
                Y_batch = Y_shuffled[:, i:i+batch_size]

                out = self.forward(X_batch)
                epoch_loss += self.loss_layer.forward(out, Y_batch) # get the loss after comparing
                num_batches += 1

                grad = self.loss_layer.backward() # backpropagate, from base case of loss layer
                for layer in reversed(self.layers):
                    grad = layer.backward(grad) # store the final gradient in the dense layers, passing through the activation layers too (activation layers like ReLU simply apply the derivative of the respective act. function and pass it onwards to the Dense layers ahead)
                
                self.optimizer.step() # make the optimizer apply all the gradients to the weights and biases of the Dense layer

            epoch_loss /= num_batches # average epoch loss in the current epoch

            if epoch % 1 == 0 or epoch == epochs -  1: # every 10 epoch or on the last epoch run a FULL forward propagation to gain insight into the full landscape on how well the network is generalizing onto the entire dataset and not just on a small batch
                out_full = self.forward(X) # forward without batching
                self.loss_layer.forward(out_full, Y)
                predictions = np.argmax(self.loss_layer.A, axis=0) # find maximum probability vertically, return 1d array
                labels = np.argmax(Y, axis=0) # find respective ONE-HOT, 1d array
                accuracy = np.mean(labels == predictions) * 100
                print(f"EPOCH: {epoch:3d} | LOSS: {epoch_loss:6f} | ACCURACY {accuracy:6f}%")

    def predict(self, X):
        out = self.forward(X)
        shift_out = out - np.max(out, axis=0, keepdims=True) # prevent overflow error in pre-act. values get too big
        return np.exp(shift_out) / (np.sum(np.exp(shift_out), axis=0, keepdims=True)) # softmax to get probabilities + broadcasting

# MODULE MODULE COMPLETE.
