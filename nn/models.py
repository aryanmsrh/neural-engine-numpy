import json
import time
import numpy as np
from .utils import softmax, accuracy


class Model:
    def forward(self, X): raise NotImplementedError
    def predict(self, X): raise NotImplementedError


class Sequential(Model):
    def __init__(self, layers, optimizer):
        self.hidden_layers = layers[:-1]
        self.loss_layer = layers[-1]
        self.optimizer = optimizer
        self.status = "idle"
        self.telemetry = {
            "start": {"total_epochs": 0, "total_steps": 0, "time": time.time()},
            "epochs": [],
            "steps": [],
            "end": {},
        }

    def forward(self, X):
        out = X
        for layer in self.hidden_layers:
            out = layer.forward(out)
        return out

    def predict(self, X):
        return softmax(self.forward(X))

    def save(self, path="model"):
        if not self.telemetry["epochs"]:
            return None
        last = self.telemetry["epochs"][-1]
        self.telemetry["end"] = {
            "type": "end",
            "epoch": last["epoch"],
            "total_epochs": self.telemetry["start"]["total_epochs"],
            "step": last["step"],
            "total_steps": self.telemetry["start"]["total_steps"],
            "loss": last["loss"],
            "accuracy": last["accuracy"],
            "elapsed_sec": last["elapsed_sec"],
            "time": time.time(),
        }
        self.save_weights(f"{path}/model_weights.npz")
        return self.save_stats(f"{path}/model_stats.json")

    save_checkpoint = save

    def stream(self, X_train, Y_train, X_test, Y_test, epochs, batch_size, path="model"):
        self.status = "running"
        start_time = time.time()
        paused_time = 0.0

        m = X_train.shape[1]
        batches = int(np.ceil(m / batch_size))
        total_steps = epochs * batches
        step = 0

        self.telemetry = {
            "start": {
                "type": "start",
                "epoch": 0,
                "total_epochs": epochs,
                "batch": 0,
                "total_batches": batches,
                "step": 0,
                "total_steps": total_steps,
                "batch_size": batch_size,
                "train_samples": X_train.shape[1],
                "test_samples": X_test.shape[1],
                "elapsed_sec": 0.0,
                "time": start_time,
            },
            "epochs": [],
            "steps": [],
            "end": {},
        }
        yield self.telemetry["start"]

        for epoch in range(epochs):
            perm = np.random.permutation(m)
            X_train_shuffled = X_train[:, perm]
            Y_train_shuffled = Y_train[:, perm]
            epoch_loss = 0.0

            for b in range(0, m, batch_size):
                while self.status == "paused":
                    t0 = time.time()
                    time.sleep(0.05)
                    paused_time += time.time() - t0

                if self.status == "stopped":
                    self.status = "idle"
                    if self.save(path):
                        yield self.telemetry["end"]
                    return

                X_batch = X_train_shuffled[:, b:b+batch_size]
                Y_batch = Y_train_shuffled[:, b:b+batch_size]

                out = self.forward(X_batch)
                loss = self.loss_layer.forward(out, Y_batch)
                epoch_loss += loss

                grad = self.loss_layer.backward()
                for layer in reversed(self.hidden_layers):
                    grad = layer.backward(grad)

                self.optimizer.step(self.hidden_layers)
                step += 1

                elapsed = round(time.time() - start_time - paused_time, 4)
                speed = round((step * batch_size) / max(0.01, elapsed))

                step_ev = {
                    "type": "step",
                    "epoch": epoch + 1,
                    "total_epochs": epochs,
                    "batch": (b // batch_size) + 1,
                    "total_batches": batches,
                    "step": step,
                    "total_steps": total_steps,
                    "loss": float(loss),
                    "accuracy": float(accuracy(softmax(out), Y_batch)),
                    "samples_per_sec": speed,
                    "elapsed_sec": elapsed,
                    "time": time.time(),
                }
                self.telemetry["steps"].append(step_ev)
                yield step_ev

            elapsed = round(time.time() - start_time - paused_time, 4)
            epoch_ev = {
                "type": "epoch",
                "epoch": epoch + 1,
                "total_epochs": epochs,
                "batch": batches,
                "total_batches": batches,
                "step": step,
                "total_steps": total_steps,
                "loss": float(epoch_loss / batches),
                "accuracy": float(accuracy(self.predict(X_test), Y_test)),
                "elapsed_sec": elapsed,
                "time": time.time(),
            }
            self.telemetry["epochs"].append(epoch_ev)
            yield epoch_ev

        self.status = "idle"
        self.save(path)
        yield self.telemetry["end"]

    def fit(self, X_train, Y_train, X_test, Y_test, epochs, batch_size, verbose=2, log_path="model/model_train_log.json", path="model"):
        RESET, DIM, YELLOW, RED = "\033[0m", "\033[2m", "\033[33m", "\033[31m"

        for event in self.stream(X_train, Y_train, X_test, Y_test, epochs, batch_size, path=path):
            etype = event["type"]

            if etype == "start":
                if verbose >= 1:
                    print(f"{RED}[{0.0:6.2f}S] STRT // SMPLS: {event['train_samples']:6} // EPOC: {epochs:6} // BTCH: {batch_size:6}{RESET}\n")
                if verbose >= 2:
                    print(f"{YELLOW}    [{0.0:6.2f}S] EPOC {0:6}/{epochs:6} | LOSS: {'-':6} | TACC: {'-':6}%{RESET}")

            elif etype == "step" and verbose >= 3:
                print(f"{DIM}       [{event['elapsed_sec']:6.2f}S] STEP {event['step']:6} (BTCH {event['batch']:6}) | LOSS: {event['loss']:6.4f} | BACC: {event['accuracy']:6.2f}%{RESET}")

            elif etype == "epoch" and verbose >= 2:
                print(f"{YELLOW}    [{event['elapsed_sec']:6.2f}S] EPOC {event['epoch']:6}/{epochs:6} | LOSS: {event['loss']:6.4f} | TACC: {event['accuracy']:6.2f}%{RESET}")

            elif etype == "end":
                if verbose >= 1:
                    print(f"\n{RED}[{event['elapsed_sec']:6.2f}S] OVER // FACC: {event['accuracy']:6.2f}% // FLOS: {event['loss']:6.4f} // DURN: {event['elapsed_sec']:6.2f}S{RESET}")

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
        return True

    def save_stats(self, path="model/model_stats.json"):
        dense = [l for l in self.hidden_layers if hasattr(l, "W")]
        end = self.telemetry["end"]

        stats = {
            "accuracy": round(end["accuracy"], 2),
            "loss": round(end["loss"], 4),
            "epochs": end["epoch"],
            "steps": end["step"],
            "training_duration_sec": end["elapsed_sec"],
            "parameters": sum(l.W.size + l.B.size for l in dense),
            "topology": [str(l.W.shape[1]) for l in dense] + [str(dense[-1].W.shape[0])],
            "optimizer": str(self.optimizer),
        }
        with open(path, "w") as f:
            json.dump(stats, f, indent=2)
        return stats