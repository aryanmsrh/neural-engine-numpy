import numpy as np
import pandas as pd


def softmax(Z, axis=0):
    shift = Z - np.max(Z, axis=axis, keepdims=True)
    return np.exp(shift) / np.sum(np.exp(shift), axis=axis, keepdims=True)


def one_hot(y, classes=10):
    Y = np.zeros((classes, y.shape[0]))
    Y[y, np.arange(y.shape[0])] = 1
    return Y


def accuracy(y_pred, y_true):
    if y_pred.ndim > 1:
        y_pred = np.argmax(y_pred, axis=0)
    if y_true.ndim > 1:
        y_true = np.argmax(y_true, axis=0)
    return float(np.mean(y_pred == y_true) * 100)


def load_mnist(data_path="data/data.csv"):
    df = pd.read_csv(data_path)
    X = df.drop("label", axis=1).values / 255.0
    Y = df["label"].values

    m = X.shape[0]
    perm = np.random.permutation(m)
    split = int(m * 0.8)

    X_train = X[perm[:split]].T
    X_test = X[perm[split:]].T
    Y_train = one_hot(Y[perm[:split]])
    Y_test = Y[perm[split:]]

    return X_train, Y_train, X_test, Y_test
