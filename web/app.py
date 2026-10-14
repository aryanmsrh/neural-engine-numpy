import sys
sys.path.extend([".", "..", "web"])

import time
import json
import numpy as np

from flask import Flask, request, jsonify, render_template, Response
from preprocess import process_canvas_image, decode_base64_image

from nn.utils import load_mnist
from nn.layers import Dense, ReLU, SoftmaxCrossEntropy
from nn.optimizers import SGD
from nn.models import Sequential

app = Flask(__name__, template_folder="templates")
app.config["TEMPLATES_AUTO_RELOAD"] = True
model = Sequential(
    layers=[
        Dense(784, 128),
        ReLU(),
        Dense(128, 64),
        ReLU(),
        Dense(64, 10),
        SoftmaxCrossEntropy(),
    ],
        optimizer=SGD(),
)
model.load_weights("model/model_weights.npz")

@app.route("/")
def index():
    with open("model/model_stats.json") as f:
        stats = json.load(f)
    return render_template("index.html", stats=stats)

@app.route("/train", methods=["GET"])
@app.route("/train/<action>", methods=["GET", "POST"])
def train_route(action="start"):
    match action:
        case "start":
            model.delete_weights()
            epochs = int(request.args.get("epochs", 10))
            lr = float(request.args.get("lr", 0.1))
            batch_size = int(request.args.get("batch_size", 64))
            model.optimizer.lr = lr

            def train():
                X_train, Y_train, X_test, Y_test = load_mnist("data/data.csv")
                for event in model.stream(X_train, Y_train, X_test, Y_test, epochs, batch_size):
                    yield f"data: {json.dumps(event)}\n\n"

            return Response(train(), mimetype="text/event-stream")

        case "pause":
            model.status = "paused"
            return jsonify({"status": "paused"})

        case "resume":
            model.status = "running"
            return jsonify({"status": "running"})

        case "stop":
            model.status = "stopped"
            stats = model.save_checkpoint()
            return jsonify({"status": "stopped", "stats": stats})

        case "save":
            stats = model.save_checkpoint()
            return jsonify({"status": "saved", "stats": stats})

        case _:
            return jsonify({"error": f"Invalid action: {action}"}), 400

@app.route("/predict", methods=["POST"])
def predict():
    img = decode_base64_image(request.json["image"])
    x, preview_b64 = process_canvas_image(img)

    t0 = time.perf_counter()
    probs = model.predict(x)
    tau_fwd = (time.perf_counter() - t0) * 1000.0

    pred = int(np.argmax(probs, axis=0)[0])
    return jsonify({
        "prediction": pred,
        "confidence": float(probs[pred, 0]),
        "probabilities": [float(p) for p in probs[:, 0]],
        "inference_time_ms": tau_fwd,
        "preview": preview_b64,
    })

if __name__ == "__main__":
    app.run()
