import sys
import json
from flask import Flask, request, jsonify, render_template, Response

sys.path.extend([".", "..", "web"]) # grab local modules

from preprocess import process_canvas_image, decode_base64_image
from inference import DigitRecognizer
from train import train

app = Flask(__name__, template_folder="templates")
recognizer = DigitRecognizer() # loads weights on startup

@app.route("/")
def index():
    stats = json.load(open("weights/model_stats.json"))
    return render_template("index.html", stats=stats)

@app.route("/architecture")
def architecture():
    return render_template("architecture.html")

@app.route("/api/status")
def status():
    stats = json.load(open("weights/model_stats.json"))
    return jsonify({
        "status": "online",
        "accuracy": stats.get("test_accuracy"),
        "loss": stats.get("loss"),
        "epochs_trained": stats.get("epochs_trained")
    })

# sse streaming endpoint for live retraining
@app.route("/api/train/stream")
def train_stream():
    epochs = int(request.args.get("epochs", 10))
    lr = float(request.args.get("lr", 0.1))

    def generate():
        for event in train(epochs=epochs, lr=lr):
            if event["type"] == "finished":
                recognizer.load_weights("weights/model_weights.npz") # reload freshly trained weights
            yield f"data: {json.dumps(event)}\n\n"

    return Response(generate(), mimetype="text/event-stream")

@app.route("/api/predict", methods=["POST"])
def predict():
    img = decode_base64_image(request.json["image"])
    x_vector, preview_b64 = process_canvas_image(img) # prep 28x28 centered vector
    result = recognizer.predict(x_vector)
    result["preview"] = preview_b64
    return jsonify(result)

if __name__ == "__main__":
    import argparse
    import socket

    parser = argparse.ArgumentParser(description="neural-engine web demo")
    parser.add_argument("-p", "--port", type=int, default=None, help="port to run web server on")
    args = parser.parse_args()

    port = args.port
    if port is None:
        port = 5000
        for p in range(5000, 5100):
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                try:
                    s.bind(("0.0.0.0", p))
                    port = p
                    break
                except OSError:
                    continue

    print(f"server running at http://localhost:{port}")
    app.run(host="0.0.0.0", port=port, debug=False)
