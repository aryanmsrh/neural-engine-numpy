import sys
import json
import numpy as np
import pandas as pd

sys.path.extend([".", ".."]) # grab nn modules

from nn.layers import Dense, ReLU, SoftmaxCrossEntropy
from nn.optimizers import SGD
from nn.models import Sequential

_cached_data = None # cache dataframe so re-training starts immediately

def save(mlp, accuracy, loss, epochs, output_path="weights/model_weights.npz"):
    weights = {f"W_{i}": l.W for i, l in enumerate(mlp.layers) if hasattr(l, "W")}
    weights.update({f"B_{i}": l.B for i, l in enumerate(mlp.layers) if hasattr(l, "B")})
    np.savez_compressed(output_path, **weights)

    # persist stats alongside weights
    stats = {
        "test_accuracy": round(accuracy, 2),
        "loss": round(loss, 4),
        "epochs_trained": epochs,
        "parameters": 109386,
        "topology": "784 -> 128 -> 64 -> 10",
        "status": "online"
    }
    json.dump(stats, open("weights/model_stats.json", "w"), indent=2)
    print(f"saved weights to {output_path} and stats to weights/model_stats.json")

def train(epochs=10, batch_size=64, lr=0.1, data_path="data/data.csv", output_path="weights/model_weights.npz"):
    global _cached_data
    if _cached_data is None:
        _cached_data = pd.read_csv(data_path)
    data = _cached_data

    X = data.drop("label", axis=1).values / 255.0 # normalize pixels to [0, 1]
    Y = data["label"].values

    m = X.shape[0]
    perm = np.random.permutation(m)
    split = int(m * 0.8) # 80-20 train-test split

    X_train, X_test = X[perm[:split]].T, X[perm[split:]].T
    Y_train_raw, Y_test_raw = Y[perm[:split]], Y[perm[split:]]

    # one-hot encode targets
    n_train = Y_train_raw.shape[0]
    Y_train = np.zeros((10, n_train))
    Y_train[Y_train_raw, np.arange(n_train)] = 1

    mlp = Sequential(
        layers=[
            Dense(784, 128),
            ReLU(),
            Dense(128, 64),
            ReLU(),
            Dense(64, 10),
            SoftmaxCrossEntropy()
        ],
        optimizer=SGD,
        learning_rate=lr
    )

    m_train = X_train.shape[1]
    total_batches = int(np.ceil(m_train / batch_size))

    yield {
        "type": "init",
        "total_epochs": epochs,
        "total_batches": total_batches,
        "train_samples": m_train,
        "test_samples": X_test.shape[1]
    }

    step_counter = 0
    final_acc = 0.0
    latest_loss = 0.0

    for epoch in range(epochs):
        epoch_perm = np.random.permutation(m_train)
        X_shuffled = X_train[:, epoch_perm]
        Y_shuffled = Y_train[:, epoch_perm]
        epoch_loss = 0.0

        for b_idx in range(0, m_train, batch_size):
            X_batch = X_shuffled[:, b_idx:b_idx+batch_size]
            Y_batch = Y_shuffled[:, b_idx:b_idx+batch_size]
            batch_num = (b_idx // batch_size) + 1

            # forward pass
            out = mlp.forward(X_batch)
            loss = mlp.loss_layer.forward(out, Y_batch)
            epoch_loss += loss
            latest_loss = float(loss)

            batch_preds = np.argmax(mlp.loss_layer.A, axis=0)
            batch_targets = np.argmax(Y_batch, axis=0)
            batch_acc = float(np.mean(batch_preds == batch_targets) * 100.0)

            # backward pass
            grad = mlp.loss_layer.backward()
            for layer in reversed(mlp.layers):
                grad = layer.backward(grad)

            # update weights
            mlp.optimizer.step()
            step_counter += 1

            # yield telemetry every 2 batches
            if batch_num % 2 == 0 or batch_num == total_batches:
                yield {
                    "type": "step",
                    "epoch": epoch + 1,
                    "total_epochs": epochs,
                    "batch": batch_num,
                    "total_batches": total_batches,
                    "step": step_counter,
                    "loss": round(float(loss), 4),
                    "batch_accuracy": round(batch_acc, 2)
                }

        # evaluate test accuracy at end of each epoch
        test_probs = mlp.predict(X_test)
        test_preds = np.argmax(test_probs, axis=0)
        final_acc = float(np.mean(test_preds == Y_test_raw) * 100.0)
        avg_loss = float(epoch_loss / total_batches)
        print(f"epoch {epoch+1:2d}/{epochs} | loss: {avg_loss:.4f} | test acc: {final_acc:.2f}%")

        yield {
            "type": "epoch_complete",
            "epoch": epoch + 1,
            "total_epochs": epochs,
            "test_accuracy": round(final_acc, 2),
            "avg_loss": round(avg_loss, 4)
        }

    # save weights and dynamic loss/accuracy to disk
    save(mlp, final_acc, latest_loss, epochs, output_path)

    yield {
        "type": "finished",
        "final_accuracy": round(final_acc, 2),
        "total_epochs": epochs,
        "final_loss": round(latest_loss, 4)
    }

if __name__ == "__main__":
    for _ in train(epochs=10):
        pass
