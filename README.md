<pre>
 _   _                      _   _____             _            
| \ | | ___ _   _ _ __ __ _| | |  ___|__ _ __ ___(_)_ __   ___ 
|  \| |/ _ \ | | | '__/ _` | | | |_ / _ \ '_ ` _ \ | '_ \ / _ \
| |\  |  __/ |_| | | | (_| | | |  _|  __/ | | | | | | | |  __/
|_| \_|\___|\__,_|_|  \__,_|_| |_|  \___|_| |_| |_|_|_| |_|\___|
 _   _                 _   _       
| \ | |_   _ _ __ ___ | | | | |  _ 
|  \| | | | | '_ ` _ \| | | | | | |
| |\  | |_| | | | | | | | | |_| | |
|_| \_|\__,_|_| |_| |_|_|  \___/|_|
</pre>

[![Python](https://img.shields.io/badge/Python-3.8%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)]()
[![NumPy](https://img.shields.io/badge/NumPy-Pure%20Matrix%20Math-013243?style=for-the-badge&logo=numpy&logoColor=white)]()
[![Frameworks](<https://img.shields.io/badge/Frameworks-Zero%20(Pure%20First%20Principles)-red?style=for-the-badge>)]()
[![License](https://img.shields.io/badge/License-MIT-blue?style=for-the-badge)]()

> A lightweight, modular deep learning framework built entirely from first principles using pure Python and NumPy. Zero black-box autograd. Zero PyTorch. Zero TensorFlow. Just raw matrix operations.

---

## [ SYSTEM.CORE_PIPELINE ]

The baseline architecture classifies 28x28 pixel handwritten digits from the MNIST dataset into 10 distinct classes (`0-9`).

```text
[INPUT] -> (784 x m) Raw Pixel Matrix
   |
   v
[DENSE] -> Z1 = W1 @ X + B1 (128 x m)
   |
   v
[RELU]  -> A1 = max(0, Z1)
   |
   v
[DENSE] -> Z2 = W2 @ A1 + B2 (64 x m)
   |
   v
[RELU]  -> A2 = max(0, Z2)
   |
   v
[DENSE] -> Z3 = W3 @ A2 + B3 (10 x m)
   |
   v
[LOSS]  -> Softmax Cross-Entropy (dZ3 = A3 - Y)
   |
   v
[SGD]   -> W = W - a*dW | B = B - a*dB
```

---

## [ DATA.MATHEMATICS ]

**Notice:** The explicit mathematical derivations (multivariate chain rules, Softmax Jacobian matrix reductions, and He weight initialization proofs) have been intentionally omitted from this README.

If you want the actual calculus and not just a high-level summary, consult the source material:

- **Video Breakdown:** [Explaining The Entire Math & Coding A Neural Engine From Scratch](https://youtu.be/KnZg2GKFDcQ)
- **Raw Math Documentation:** `NEURAL ENGINE.pdf` (Included in this repo).

---

## [ DIR.STRUCTURE ]

```text
neural-engine-numpy/
├── main.py                :: MNIST loader & training loop execution
├── NEURAL ENGINE.pdf      :: Handwritten math derivations & notes
├── data.csv               :: MNIST dataset (60,000 samples)
├── LICENSE                :: MIT License
└── nn/                    :: Core Engine Architecture
    ├── __init__.py        :: Package exports
    ├── layers.py          :: Dense, ReLU, SoftmaxCrossEntropy
    ├── models.py          :: Sequential container
    └── optimizers.py      :: SGD parameter updates
```

---

## [ EXECUTION.ENVIRONMENT ]

Ensure your local environment has the standard data stack installed:

```bash
pip install numpy pandas
```

Initialize the training sequence:

```bash
python main.py
```

### Custom Implementation

```python
import numpy as np
from nn import Sequential, Dense, ReLU, SoftmaxCrossEntropy, SGD

# Initialize architecture
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

# Execute training block
model.fit(X_train, Y_train, epochs=10, batch_size=64)

# Inference
probs = model.predict(X_test)
predictions = np.argmax(probs, axis=0)
```

---

## [ SYS.BENCHMARK ]

Training a `784 -> 128 -> 64 -> 10` architecture with `batch_size=64` and `alpha=0.1` on pure NumPy:

| Epoch | Training Loss | Training Acc | Test Acc |
| :---: | :-----------: | :----------: | :------: |
|   0   |   0.354731    |    92.98%    |    —     |
|   5   |   0.064004    |    98.65%    |    —     |
|   9   |   0.033100    |    99.43%    |  97.29%  |

---

## [ FUTURE.ROADMAP ]

- **Optimizers:** Adam, RMSProp, SGD with Momentum.
- **Regularization:** Inverted Dropout, Batch Normalization, L2 Weight Decay.
- **Activations:** LeakyReLU, Tanh.
- **Diagnostics:** Finite-Difference Numerical Gradient Checking (`gradcheck`).

---

## [ LIC.AUTH ]

Code by Aryan Mishra ([@modestpenguinn](https://youtube.com/@modestpenguinn)).
Released under the MIT License.
