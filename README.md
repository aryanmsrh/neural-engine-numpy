# Neural Engine NumPy

![Python](https://img.shields.io/badge/Python-3.8%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white)

A deep learning framework built entirely from first principles using pure Python and NumPy. No black-box autograd, no PyTorch, no TensorFlow, no scikit-learn.

## Resources

- **Walkthrough:** [YouTube Video](https://youtu.be/KnZg2GKFDcQ)
- **Calculus Derivations:** Reference [`docs/NEURAL ENGINE.pdf`](docs/NEURAL%20ENGINE.pdf) in this repository for the full backpropagation and optimization proofs.

## Project Structure

```text
neural-engine-numpy/
├── data/              # Raw datasets (e.g. data.csv tracked via Git LFS)
├── docs/              # Mathematical derivations and calculus notes (NEURAL ENGINE.pdf)
├── model/             # Persisted weight checkpoints and stats (model_weights.npz, model_stats.json)
├── nn/                # Core pure NumPy neural network framework
│   ├── layers.py      # Dense, ReLU, SoftmaxCrossEntropy
│   ├── models.py      # Sequential container with stream() and fit()
│   ├── optimizers.py  # SGD optimizer
│   └── utils.py       # Numerically stable softmax, one_hot, accuracy, load_mnist
├── scripts/           # Training and evaluation entry points
│   └── test.py        # CLI training and evaluation script
└── web/               # Interactive web app for digit drawing & live training
    ├── app.py         # Flask server with /predict and /train SSE stream
    ├── preprocess.py  # Drawing canvas 28x28 preprocessor & center-of-mass alignment
    ├── static/        # Frontend CSS, modular JS (predict.js, canvas.js, train.js), fonts
    └── templates/     # UI templates (index.html, base.html)
```

## Architecture

MNIST baseline model:
`Input (784) -> Dense(128) -> ReLU -> Dense(64) -> ReLU -> Dense(10) -> Softmax Cross-Entropy`

- **Weights Init:** He Normal $\mathcal{N}(0, \sqrt{2 / d_{\text{in}}})$
- **Biases Init:** Zero vectors $(d_{\text{out}}, 1)$
- **Stability:** Column-wise max-logit subtraction before `exp` in Softmax; probability clipping with $\epsilon = 10^{-15}$ in cross-entropy loss.

## Quick Start

**Pull Dataset (Git LFS):**

The dataset (`data/data.csv`) is tracked via [Git LFS](https://git-lfs.com/). If you don't have Git LFS installed yet:

- **macOS:**
  ```bash
  brew install git-lfs
  ```
- **Linux:**
  ```bash
  sudo apt install git-lfs      # Ubuntu / Debian
  sudo dnf install git-lfs      # Fedora / RHEL
  sudo pacman -S git-lfs        # Arch Linux
  ```
- **Windows:**
  ```powershell
  winget install GitHub.GitLFS  # or choco install git-lfs
  ```

Once installed, pull the full dataset:

```bash
git lfs install && git lfs pull
```

**Dependencies:**

```bash
pip install -r requirements.txt
```

**Run Interactive Web Demo:**

```bash
python web/app.py
```

Open the printed URL (`http://127.0.0.1:5000`) in your browser. The web interface features:

- **Interactive Drawing Canvas:** 280×280 interactive canvas with dynamic brush strokes, live MNIST centering, and real-time inference latency $\tau_{\text{fwd}}$.
- **Live Training Studio:** Real-time streaming training over Server-Sent Events (`/train`) with weight purging, dynamic loss and accuracy curves, and terminal-style verbose stdout logs.

## CLI Usage & Custom Models

**Run CLI Training:**

```bash
python scripts/test.py
```

**Custom Implementation:**

```python
import numpy as np
from nn.layers import Dense, ReLU, SoftmaxCrossEntropy
from nn.optimizers import SGD
from nn.models import Sequential
from nn.utils import load_mnist

# 1. Load dataset (80/20 train/test split)
X_train, Y_train, X_test, Y_test = load_mnist("data/data.csv")

# 2. Build model
model = Sequential(
    layers=[
        Dense(784, 128),
        ReLU(),
        Dense(128, 64),
        ReLU(),
        Dense(64, 10),
        SoftmaxCrossEntropy()
    ],
    optimizer=SGD(lr=0.1)
)

# 3. Train synchronously (or stream events via model.stream)
telemetry = model.fit(
    X_train, Y_train, X_test, Y_test,
    epochs=10,
    batch_size=64,
    verbose=2
)

# 4. Save checkpoints and inference
model.save_weights("model/model_weights.npz")
model.save_stats(telemetry, "model/model_stats.json")

probs = model.predict(X_test)
```

## Benchmark

_Batch Size: 64 | Learning Rate: 0.1 | Optimizer: SGD_

| Epoch | Training Loss | Training Acc | Test Acc |
| :---: | :-----------: | :----------: | :------: |
|   1   |    0.3547     |    92.98%    |  94.12%  |
|   5   |    0.0640     |    98.65%    |  96.87%  |
|  10   |    0.0331     |    99.43%    |  97.29%  |

## To Implement

As pointed out in the [video breakdown](https://youtu.be/KnZg2GKFDcQ), the current architecture can be further enhanced with:

**Advanced Optimizers**

- **Adam:** (Adaptive Moment Estimation) Dynamic per-parameter learning rates with first and second moment tracking.
- **RMSProp:** Root Mean Square Propagation to prevent exploding/vanishing gradients.
- **SGD + Momentum:** Adding velocity vectors to accelerate past saddle points and local minima.

**Regularization**

- **Dropout:** Zeroing out random neuron subsets during training forward passes to reduce co-adaptation.
- **Early Stopping:** Halting training when validation loss stops improving.

**Alternative Activations**

- **LeakyReLU:** Prevents dying ReLU by allowing a small non-zero gradient for negative pre-activations.
- **Tanh:** Zero-centered hyperbolic tangent activation.

## License

MIT
