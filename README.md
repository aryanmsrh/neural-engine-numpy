# Neural Engine NumPy

![Python](https://img.shields.io/badge/Python-3.8%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white)

A deep learning framework built entirely from first principles using pure Python and NumPy. No black-box autograd, no PyTorch, no TensorFlow.

## Resources

- **Walkthrough:** [YouTube Video](https://youtu.be/KnZg2GKFDcQ)
- **Calculus Derivations:** Reference [`docs/NEURAL ENGINE.pdf`](docs/NEURAL%20ENGINE.pdf) in this repository for the full backpropagation and optimization proofs.

## Project Structure

```text
neural-engine-numpy/
├── data/              # Raw datasets (e.g. data.csv)
├── docs/              # Mathematical derivations and calculus notes (NEURAL ENGINE.pdf)
├── nn/                # Core pure NumPy neural network framework
│   ├── __init__.py
│   ├── layers.py      # Dense, ReLU, SoftmaxCrossEntropy
│   ├── models.py      # Sequential model container
│   └── optimizers.py  # SGD optimizer
├── scripts/           # Training and evaluation entry points
│   └── main.py        # 10-epoch training and evaluation script
├── web/               # Interactive web demo for handwritten digit recognition
│   ├── app.py         # Flask web server
│   ├── train.py       # Training pipeline & streaming engine
│   ├── inference.py   # Pure NumPy inference engine
│   ├── preprocess.py  # Canvas image preprocessor
│   └── templates/     # UI templates (index.html, architecture.html)
└── weights/           # Trained weight checkpoints (model_weights.npz)
```

## Architecture

MNIST baseline model:
`Input (784) -> Dense(128) -> ReLU -> Dense(64) -> ReLU -> Dense(10) -> Softmax Cross-Entropy`

## Quick Start

**Dependencies:**

```bash
pip install -r requirements.txt
```

**Run Training / Evaluation:**

```bash
python scripts/main.py
```

Or to train and save checkpoints to `weights/model_weights.npz`:

```bash
python web/train.py
```

**Run Interactive Web Demo:**

```bash
python web/app.py
```

Open `http://localhost:5000` in your browser.

**Custom Implementation:**

```python
import numpy as np
from nn import Sequential, Dense, ReLU, SoftmaxCrossEntropy, SGD

model = Sequential(
    layers=[
        Dense(input_dim=784, output_dim=128),
        ReLU(),
        Dense(input_dim=128, output_dim=64),
        ReLU(),
        Dense(input_dim=64, output_dim=10),
        SoftmaxCrossEntropy()
    ],
    optimizer=SGD,
    learning_rate=0.1
)

model.fit(X_train, Y_train, epochs=10, batch_size=64)
probs = model.predict(X_test)
```

## Benchmark

_Batch Size: 64 | Learning Rate: 0.1 | Optimizer: SGD_

| Epoch | Training Loss | Training Acc | Test Acc |
| :---: | :-----------: | :----------: | :------: |
|   0   |   0.354731    |    92.98%    |    —     |
|   5   |   0.064004    |    98.65%    |    —     |
|   9   |   0.033100    |    99.43%    |  97.29%  |

## To Implement

As pointed out in the [video breakdown](https://youtu.be/KnZg2GKFDcQ), the current architecture suffers from overfitting. The network possesses enough capacity to memorize the training data, leading to a gap between training and test accuracy. To make this engine production-ready, the following upgrades are pending:

**Advanced Optimizers**

- **Adam:** (Adaptive Moment Estimation) The current industry standard for dynamic learning rates.
- **RMSProp:** (Root Mean Square Propagation) Adapts the learning rate by dividing the gradient by a running average of its recent magnitude. Prevents gradients from exploding or vanishing.
- **SGD + Momentum:** Adds velocity to the gradient steps to push through local minima and accelerate convergence.

**Regularization**

- **Dropout Layers:** Randomly zeroing out a percentage of neurons during the forward pass to force the network to learn generalized features instead of memorizing specific pixel paths.
- **Early Stopping:** Halting the training loop the exact epoch test accuracy begins to diverge from training accuracy.

**Alternative Activations**

- **LeakyReLU:** Solves the "dying ReLU" problem by allowing a small, non-zero gradient when pre-activations are negative, keeping neurons alive.
- **Tanh:** An alternative zero-centered activation function.

## License

MIT
