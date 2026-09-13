import sys
import time
import numpy as np

sys.path.extend([".", ".."]) # grab nn modules

from nn.layers import Dense, ReLU, SoftmaxCrossEntropy
from nn.optimizers import SGD
from nn.models import Sequential

class DigitRecognizer:
    def __init__(self, weights_path="weights/model_weights.npz"):
        self.model = Sequential(
            layers=[
                Dense(784, 128),
                ReLU(),
                Dense(128, 64),
                ReLU(),
                Dense(64, 10),
                SoftmaxCrossEntropy()
            ],
            optimizer=SGD,
            learning_rate=0.1
        )
        self.load_weights(weights_path)

    def load_weights(self, path):
        data = np.load(path)
        for i, layer in enumerate(self.model.layers):
            if hasattr(layer, "W"):
                layer.W = np.array(data[f"W_{i}"], copy=True)
                layer.B = np.array(data[f"B_{i}"], copy=True)
        print(f"loaded weights from {path}")

    def predict(self, x_vector):
        if x_vector.ndim == 1:
            x_vector = x_vector.reshape(-1, 1) # (784, 1)

        t0 = time.perf_counter()
        probs = self.model.predict(x_vector) # forward pass + stable softmax
        inference_time_ms = (time.perf_counter() - t0) * 1000.0

        predicted_class = int(np.argmax(probs, axis=0)[0])
        confidence = float(probs[predicted_class, 0])
        all_probs = [float(p) for p in probs[:, 0]]

        return {
            "prediction": predicted_class,
            "confidence": confidence,
            "probabilities": all_probs,
            "inference_time_ms": round(inference_time_ms, 3)
        }
