# train a network on the mnist database

import numpy as np
from nn.layers import Dense, ReLU, SoftmaxCrossEntropy
from nn.optimizers import SGD
from nn.models import Sequential

import pandas as pd # for loading mnist dataset

np.random.seed(42) # for reproducibility

print("LOADING MNIST")
data = pd.read_csv("data.csv")

X = data.drop('label', axis=1).values / 255.0 # (60000, 784)
Y = data['label'].values

m = X.shape[0] # number of samples (60000)
perm = np.random.permutation(m) # shuffle dataset
split = int(m * 0.8) # 80-20 train-test split

X_train, X_test = X[perm[:split]], X[perm[split:]]
Y_train, Y_test = Y[perm[:split]], Y[perm[split:]]

X_train = X_train.T # (784, 48000)
X_test = X_test.T # (784, 12000)

def one_hot_enc(y, num_classes=10):
    m = y.shape[0]
    Y = np.zeros((num_classes, m))
    Y[y, np.arange(m)] = 1 # y gives row, m gives column
    return Y

Y_train = one_hot_enc(Y_train)
Y_test = one_hot_enc(Y_test)

mlp = Sequential(
    layers = [
        Dense(input_dim=784, output_dim=128),
        ReLU(),
        Dense(input_dim=128, output_dim=64),
        ReLU(),
        Dense(input_dim=64, output_dim=10),
        SoftmaxCrossEntropy()
    ],
    optimizer=SGD,
    learning_rate=0.1
) # multi-layer perceptron to train on digit dataset

mlp.fit(X_train, Y_train, epochs=10, batch_size=64) # train model weights and biases

print("\n=== TEST SET EVALUATION ===\n")
predictions = np.argmax(mlp.predict(X_test), axis=0)
labels = np.argmax(Y_test, axis=0)
accuracy = np.mean(predictions == labels) * 100
print(f"ACCURACY ON TEST SET: {accuracy:.6f}%")

# mnist training complete
