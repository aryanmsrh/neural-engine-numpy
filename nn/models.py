import json
import numpy as np
from .utils import softmax, accuracy
import time


class Model:
    def forward(self, X): raise NotImplementedError
    def predict(self, X): raise NotImplementedError

class Sequential(Model):
    def __init__(self, layers, optimizer):
        self.hidden_layers = layers[:-1]
        self.loss_layer = layers[-1]
        self.optimizer = optimizer

    def forward(self, X):
        out = X
        for hidden_layer in self.hidden_layers:
            out = hidden_layer.forward(out)
        return out

    def predict(self, X):
        return softmax(self.forward(X))

    def stream(self, X_train, Y_train, X_test, Y_test, epochs, batch_size):
        self.telemetry = {
            "start": {},
            "epochs": [],
            "steps": [],
            "end": {}
        }
        
        m = X_train.shape[1]
        batches = int(np.ceil(m / batch_size))
        step = 0
        test_acc, avg_loss = 0, 0

        start_ev = {
            "type": "start",
            "total_epochs": epochs,
            "total_batches": batches,
            "train_samples": X_train.shape[1],
            "test_samples": X_test.shape[1],
            "time": time.time(),
        }
        self.telemetry["start"] = start_ev
        yield start_ev

        for epoch in range(epochs):
            perm = np.random.permutation(m)
            X_train_shuffled = X_train[:, perm]
            Y_train_shuffled = Y_train[:, perm]
            epoch_loss = 0

            for b in range(0, m, batch_size):
                X_train_shuffled_batch = X_train_shuffled[:, b:b+batch_size]
                Y_train_shuffled_batch = Y_train_shuffled[:, b:b+batch_size]
                batch_num = (b // batch_size) + 1

                out = self.forward(X_train_shuffled_batch)
                loss = self.loss_layer.forward(out, Y_train_shuffled_batch)
                epoch_loss += loss

                grad = self.loss_layer.backward()
                for hidden_layer in reversed(self.hidden_layers):
                    grad = hidden_layer.backward(grad)

                self.optimizer.step(self.hidden_layers)
                step += 1

                batch_acc = accuracy(softmax(out), Y_train_shuffled_batch)

                step_ev = {
                    "type": "step",
                    "step": step,
                    "batch": batch_num,
                    "loss": float(loss),
                    "batch_accuracy": float(batch_acc),
                    "time": time.time(),
                }
                self.telemetry["steps"].append(step_ev)
                yield step_ev

            test_acc = float(accuracy(self.predict(X_test), Y_test))
            avg_loss = float(epoch_loss / batches)

            epoch_ev = {
                "type": "epoch",
                "epoch": epoch + 1,
                "test_accuracy": test_acc,
                "avg_loss": avg_loss,
                "time": time.time()
            }
            self.telemetry["epochs"].append(epoch_ev)
            yield epoch_ev

        end_ev = {
            "type": "end",
            "final_accuracy": test_acc,
            "total_epochs": epochs,
            "final_loss": avg_loss,
            "time": time.time()
        }
        self.telemetry["end"] = end_ev
        yield end_ev

        self.save_weights()
        self.save_stats(self.telemetry)

    def fit(self, X_train, Y_train, X_test, Y_test, epochs, batch_size, verbose=2, log_path="model/model_train_log.json"):
        start_time = 0.0
        RESET, DIM, YELLOW, RED = "\033[0m", "\033[2m", "\033[33m", "\033[31m"

        for event in self.stream(X_train, Y_train, X_test, Y_test, epochs, batch_size):
            etype = event["type"]

            if etype == "start":
                start_time = event["time"]
                if verbose >= 1:
                    print(f"{RED}[{0:6.2f}S] STRT // SMPLS: {event['train_samples']:6} // EPOC: {epochs:6} // BTCH: {batch_size:6}{RESET}\n")
                if verbose >= 2:
                    print(f"{YELLOW}    [{0:6.2f}S] EPOC {0:6}/{epochs:6} | LOSS: {'-':6} | TACC: {'-':6}%{RESET}")

            elif etype == "step" and verbose >= 3:
                elapsed = event["time"] - start_time
                print(f"{DIM}       [{elapsed:6.2f}S] STEP {event['step']:6} (BTCH {event['batch']:6}) | LOSS: {event['loss']:6.4f} | BACC: {event['batch_accuracy']:6.2f}%{RESET}")

            elif etype == "epoch" and verbose >= 2:
                elapsed = event["time"] - start_time
                print(f"{YELLOW}    [{elapsed:6.2f}S] EPOC {event['epoch']:6}/{epochs:6} | LOSS: {event['avg_loss']:6.4f} | TACC: {event['test_accuracy']:6.2f}%{RESET}")

            elif etype == "end":
                if verbose >= 1:
                    elapsed = event["time"] - start_time
                    print(f"\n{RED}[{elapsed:6.2f}S] OVER // FACC: {event['final_accuracy']:6.2f}% // FLOS: {event['final_loss']:6.4f} // DURN: {self.telemetry['end']['time']-self.telemetry['start']['time']:6.2f}S{RESET}")

                if log_path:
                    with open(log_path, "w") as f:
                        json.dump(self.telemetry, f, indent=2)

        return self.telemetry

    def save_weights(self, path="model/model_weights.npz"):
        weights = {}
        for i, l in enumerate(self.hidden_layers):
            if hasattr(l, "W"):
                weights[f"W_{i}"] = l.W
                weights[f"B_{i}"] = l.B
        np.savez_compressed(path, **weights)
        return weights

    def load_weights(self, path="model/model_weights.npz"):
        data = np.load(path)
        for i, l in enumerate(self.hidden_layers):
            if hasattr(l, "W"):
                l.W = data[f"W_{i}"]
                l.B = data[f"B_{i}"]
        return True

    def delete_weights(self):
        for l in self.hidden_layers:
            if hasattr(l, "W"):
                output_dim, input_dim = l.W.shape
                l.W = np.random.randn(output_dim, input_dim) * np.sqrt(2.0 / input_dim)
                l.B = np.zeros((output_dim, 1))

    def save_stats(self, telemetry=None, path="model/model_stats.json"):
            if telemetry is None:
                telemetry = self.telemetry
            dense = [l for l in self.hidden_layers if hasattr(l, "W")]
            params = sum(l.W.size + l.B.size for l in dense)
            dims = [str(l.W.shape[1]) for l in dense] + [str(dense[-1].W.shape[0])]

            end = telemetry["end"]
            start = telemetry["start"]
            duration = end["time"] - start["time"]

            stats = {
                "test_accuracy": round(end["final_accuracy"], 2),
                "loss": round(end["final_loss"], 4),
                "final_loss": round(end["final_loss"], 4),
                "epochs": end["total_epochs"],
                "epochs_trained": end["total_epochs"],
                "training_duration_sec": round(duration, 4),
                "parameters": params,
                "topology": " -> ".join(dims),
                "status": "trained",
            }
            with open(path, "w") as f:
                json.dump(stats, f, indent=2)
            return stats