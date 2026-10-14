import sys
sys.path.extend([".", ".."])

from nn.utils import load_mnist
from nn.layers import Dense, ReLU, SoftmaxCrossEntropy
from nn.optimizers import SGD
from nn.models import Sequential

X_train, Y_train, X_test, Y_test = load_mnist("data/data.csv")

mlp = Sequential(
    layers=[
        Dense(784, 128),
        ReLU(),
        Dense(128, 64),
        ReLU(),
        Dense(64, 10),
        SoftmaxCrossEntropy()
    ],
    optimizer=SGD(lr=0.1),
)

mlp.fit(X_train, Y_train, X_test, Y_test, epochs=5, batch_size=64, verbose=3)