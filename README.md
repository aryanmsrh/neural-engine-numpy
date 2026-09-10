# Neural Engine NumPy

![Python](https://img.shields.io/badge/Python-3.8%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-Pure%20Matrix%20Math-013243?style=for-the-badge&logo=numpy&logoColor=white)
![Frameworks](<https://img.shields.io/badge/Frameworks-Zero%20(Pure%20First%20Principles)-red?style=for-the-badge>)
![License](https://img.shields.io/badge/License-MIT-blue?style=for-the-badge)

A lightweight, modular deep learning framework built from first principles using pure Python and NumPy—no PyTorch, no TensorFlow, no Autograd.

---

### Watch the Complete Math & Code Walkthrough

[![YouTube Video](https://img.youtube.com/vi/KnZg2GKFDcQ/maxresdefault.jpg)](https://youtu.be/KnZg2GKFDcQ)

_Click the banner above to watch the full step-by-step mathematical derivation and code walkthrough video on YouTube._

---

## Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Architecture & Neural Pipeline](#architecture--neural-pipeline)
- [Comprehensive Mathematical Foundations](#comprehensive-mathematical-foundations)
  - [1. Matrix Notation & Tensor Shape Invariants](#1-matrix-notation--tensor-shape-invariants)
  - [2. He (Kaiming) Weight Initialization](#2-he-kaiming-weight-initialization)
  - [3. Forward Propagation Equations](#3-forward-propagation-equations)
  - [4. Categorical Cross-Entropy Loss](#4-categorical-cross-entropy-loss)
  - [5. Analytical Backpropagation Proof](#5-analytical-backpropagation-proof)
    - [Step 5.1: Fused Softmax + Cross-Entropy Loss Gradient](#step-51-fused-softmax--cross-entropy-loss-gradient)
    - [Step 5.2: Output Layer Parameter Gradients](#step-52-output-layer-parameter-gradients)
    - [Step 5.3: Propagating Error to Hidden Layers](#step-53-propagating-error-to-hidden-layers)
  - [6. Summary of Generalized Recursion & SGD Update](#6-summary-of-generalized-recursion--sgd-update)
- [Codebase Structure](#codebase-structure)
- [Getting Started](#getting-started)
- [MNIST Benchmark & Training Results](#mnist-benchmark--training-results)
- [Future Roadmap](#future-roadmap)
- [License](#license)

---

## Overview

**Neural Engine NumPy** is a deep learning engine built strictly from first principles using Python and NumPy. It eliminates black-box automatic differentiation engines in favor of explicit analytical matrix calculus.

Every tensor operation, weight initialization, activation function, loss computation, gradient propagation, and parameter update is derived on paper and translated directly into fully vectorized NumPy matrix operations (`np.dot` / `@`).

---

## Key Features

- **Zero External ML Libraries**: Built exclusively with Python standard library and NumPy.
- **First-Principles Calculus**: Complete mathematical derivations using the multivariate chain rule and Softmax Jacobian matrix reductions.
- **Fully Vectorized**: Computes transformations across $m$ training instances in parallel without Python `for` loops over batch items.
- **Numerical Stability Guards**:
  - Softmax max-logit subtraction ($Z_{\text{shifted}} = Z - \max(Z)$) to prevent exponential overflow.
  - Probability clipping ($\text{clip}(A, 10^{-15}, 1 - 10^{-15})$) to prevent $\ln(0)$ underflow.
  - He (Kaiming) normal weight initialization tailored for ReLU activations.
- **Clean Modular API**: Object-oriented abstractions mimicking PyTorch: `Dense`, `ReLU`, `SoftmaxCrossEntropy`, `SGD`, and `Sequential`.

---

## Architecture & Neural Pipeline

The baseline implementation classifies $28 \times 28$ pixel handwritten digits from the **MNIST** dataset into 10 classes (`0-9`):

```text
       [ Input Layer: X ]
      (784 x m batch matrix)
                │
                ▼
      ┌──────────────────┐
      │  Dense Layer [1] │  ───►  Z¹ = W¹ · X + B¹   (128 x m)
      └─────────┬────────┘
                │
                ▼
      ┌──────────────────┐
      │  ReLU Activation │  ───►  A¹ = max(0, Z¹)    (128 x m)
      └─────────┬────────┘
                │
                ▼
      ┌──────────────────┐
      │  Dense Layer [2] │  ───►  Z² = W² · A¹ + B²  (64 x m)
      └─────────┬────────┘
                │
                ▼
      ┌──────────────────┐
      │  ReLU Activation │  ───►  A² = max(0, Z²)    (64 x m)
      └─────────┬────────┘
                │
                ▼
      ┌──────────────────┐
      │  Dense Layer [3] │  ───►  Z³ = W³ · A² + B³  (10 x m)
      └─────────┬────────┘
                │
                ▼
  ┌───────────────────────────┐
  │   SoftmaxCrossEntropy     │  ───►  A³ = Softmax(Z³)
  │   (Loss & Backward Base)  │  ───►  dZ³ = A³ - Y      (10 x m)
  └─────────────┬─────────────┘
                │
                ▼
      ┌──────────────────┐
      │  SGD Optimizer   │  ───►  W ◄── W - α · dW
      │  (Parameter Step)│  ───►  B ◄── B - α · dB
      └──────────────────┘
```

---

## Comprehensive Mathematical Foundations

---

### 1. Matrix Notation & Tensor Shape Invariants

We adopt a **column-vector layout** where each column in a matrix represents a single training sample $(i) \in \{1, \dots, m\}$:

- **Batch Input ($X$):** $X \in \mathbb{R}^{n_0 \times m}$ ($n_0 = 784$ for MNIST, $m$ = batch size).
- **Layer Weights ($W^{[l]}$):** $W^{[l]} \in \mathbb{R}^{n_l \times n_{l-1}}$, where $n_l$ is the number of neurons in layer $l$.
- **Layer Biases ($B^{[l]}$):** $B^{[l]} \in \mathbb{R}^{n_l \times 1}$, broadcast column-wise across $m$ samples.
- **Pre-activations ($Z^{[l]}$):** $Z^{[l]} \in \mathbb{R}^{n_l \times m}$.
- **Activations ($A^{[l]}$):** $A^{[l]} \in \mathbb{R}^{n_l \times m}$ ($A^{[0]} = X$).
- **One-Hot Targets ($Y$):** $Y \in \mathbb{R}^{n_L \times m}$ ($n_L = 10$ classes).

---

### 2. He (Kaiming) Weight Initialization

We use **He Normal Initialization** to set initial parameter weights:

$$W^{[l]} \sim \mathcal{N}\left(0, \, \sqrt{\frac{2}{n_{l-1}}}\right)$$

Because non-symmetrical activation functions like ReLU zero out all negative pre-activations (effectively killing half the signal variance), standard Xavier initialization causes signal collapse in deeper networks. He initialization scales the initial weight variance by $\sqrt{2 / n_{l-1}}$, keeping activation variances stable across all hidden layers during training.

---

### 3. Forward Propagation Equations

For layer $l \in \{1, \dots, L\}$:

#### Affine Step

$$Z^{[l]} = W^{[l]} A^{[l-1]} + B^{[l]}$$

where $W^{[l]} \in \mathbb{R}^{n_l \times n_{l-1}}$, $A^{[l-1]} \in \mathbb{R}^{n_{l-1} \times m}$, and $B^{[l]} \in \mathbb{R}^{n_l \times 1}$.

#### Hidden Layer Activation (ReLU)

$$A^{[l]} = g(Z^{[l]}) = \max(0, \, Z^{[l]})$$

#### Output Layer Activation (Softmax)

To prevent exponential overflow, subtract $\max(Z^{(i)})$ along the class dimension:

$$\hat{z}_j^{[L](i)} = z_j^{[L](i)} - \max_{r} z_r^{[L](i)}$$

$$a_j^{[L](i)} = \frac{e^{\hat{z}_j^{[L](i)}}}{\sum_{r=1}^{n_L} e^{\hat{z}_r^{[L](i)}}}$$

---

### 4. Categorical Cross-Entropy Loss

For a single sample $(i)$ with one-hot label vector $y^{(i)}$:

$$\mathcal{L}^{(i)} = -\sum_{j=1}^{n_L} y_j^{(i)} \ln\left(a_j^{[L](i)}\right)$$

Average loss across a mini-batch of size $m$:

$$\mathcal{L} = \frac{1}{m} \sum_{i=1}^{m} \mathcal{L}^{(i)} = -\frac{1}{m} \sum_{i=1}^{m} \sum_{j=1}^{n_L} y_j^{(i)} \ln\left(a_j^{[L](i)}\right)$$

---

### 5. Analytical Backpropagation Proof

#### Step 5.1: Fused Softmax + Cross-Entropy Loss Gradient

We compute $\frac{\partial \mathcal{L}^{(i)}}{\partial z_j^{[L](i)}}$. Since pre-activation $z_j^{[L]}$ influences the denominator of every output activation $a_k^{[L]}$, we apply the multivariable chain rule over all output neurons $k \in \{1, \dots, n_L\}$:

$$\frac{\partial \mathcal{L}^{(i)}}{\partial z_j^{[L](i)}} = \sum_{k=1}^{n_L} \frac{\partial \mathcal{L}^{(i)}}{\partial a_k^{[L](i)}} \frac{\partial a_k^{[L](i)}}{\partial z_j^{[L](i)}}$$

##### 1. Derivative of Loss w.r.t. Activation

$$\frac{\partial \mathcal{L}^{(i)}}{\partial a_k^{[L](i)}} = -\frac{y_k^{(i)}}{a_k^{[L](i)}}$$

##### 2. Softmax Jacobian Matrix (Quotient Rule)

Using $a_k = \frac{e^{z_k}}{S}$ where $S = \sum_{r} e^{z_r}$:

Direct Path ($k = j$):

$$\frac{\partial a_j}{\partial z_j} = \frac{e^{z_j} S - e^{z_j} e^{z_j}}{S^2} = a_j (1 - a_j)$$

Indirect Path ($k \neq j$):

$$\frac{\partial a_k}{\partial z_j} = \frac{0 \cdot S - e^{z_k} e^{z_j}}{S^2} = -a_k a_j$$

##### 3. Expanding and Collapsing the Chain Rule Sum

$$\begin{aligned} \frac{\partial \mathcal{L}^{(i)}}{\partial z_j^{[L](i)}} &= \left(-\frac{y_j^{(i)}}{a_j^{[L](i)}}\right) a_j^{[L](i)}(1 - a_j^{[L](i)}) + \sum_{k \neq j}^{n_L} \left(-\frac{y_k^{(i)}}{a_k^{[L](i)}}\right) \left(-a_k^{[L](i)} a_j^{[L](i)}\right) \\ &= -y_j^{(i)} (1 - a_j^{[L](i)}) + \sum_{k \neq j}^{n_L} y_k^{(i)} a_j^{[L](i)} \\ &= -y_j^{(i)} + y_j^{(i)} a_j^{[L](i)} + a_j^{[L](i)} \sum_{k \neq j}^{n_L} y_k^{(i)} \\ &= -y_j^{(i)} + a_j^{[L](i)} \left( y_j^{(i)} + \sum_{k \neq j}^{n_L} y_k^{(i)} \right) \\ &= -y_j^{(i)} + a_j^{[L](i)} \left( \sum_{k=1}^{n_L} y_k^{(i)} \right) \end{aligned}$$

Because $Y$ is a one-hot distribution ($\sum_{k=1}^{n_L} y_k^{(i)} = 1$):

$$\frac{\partial \mathcal{L}^{(i)}}{\partial z_j^{[L](i)}} = a_j^{[L](i)} - y_j^{(i)}$$

Vectorized across all classes and $m$ batch samples:

$$dZ^{[L]} = A^{[L]} - Y \in \mathbb{R}^{n_L \times m}$$

---

#### Step 5.2: Output Layer Parameter Gradients

Applying the chain rule through $Z^{[L]} = W^{[L]} A^{[L-1]} + B^{[L]}$:

$$\frac{\partial \mathcal{L}^{(i)}}{\partial w_{jk}^{[L]}} = \frac{\partial \mathcal{L}^{(i)}}{\partial z_j^{[L](i)}} \frac{\partial z_j^{[L](i)}}{\partial w_{jk}^{[L]}} = dZ_j^{[L](i)} a_k^{[L-1](i)}$$

Averaging across mini-batch size $m$:

$$dW^{[L]} = \frac{1}{m} dZ^{[L]} \left(A^{[L-1]}\right)^T \in \mathbb{R}^{n_L \times n_{L-1}}$$

$$dB^{[L]} = \frac{1}{m} \sum_{i=1}^{m} dZ^{[L](i)} = \frac{1}{m} \text{np.sum}(dZ^{[L]}, \text{axis}=1, \text{keepdims}=\text{True}) \in \mathbb{R}^{n_L \times 1}$$

---

#### Step 5.3: Propagating Error to Hidden Layers

To push gradient back from layer $l+1$ to layer $l$:

$$\frac{\partial \mathcal{L}^{(i)}}{\partial z_k^{[l](i)}} = \sum_{j=1}^{n_{l+1}} \frac{\partial \mathcal{L}^{(i)}}{\partial z_j^{[l+1](i)}} \frac{\partial z_j^{[l+1](i)}}{\partial a_k^{[l](i)}} \frac{\partial a_k^{[l](i)}}{\partial z_k^{[l](i)}}$$

Using:

- $\frac{\partial \mathcal{L}^{(i)}}{\partial z_j^{[l+1](i)}} = dZ_j^{[l+1](i)}$
- $\frac{\partial z_j^{[l+1](i)}}{\partial a_k^{[l](i)}} = W_{jk}^{[l+1]}$
- $\frac{\partial a_k^{[l](i)}}{\partial z_k^{[l](i)}} = g'(z_k^{[l](i)})$

$$\frac{\partial \mathcal{L}^{(i)}}{\partial z_k^{[l](i)}} = g'\left(z_k^{[l](i)}\right) \sum_{j=1}^{n_{l+1}} W_{jk}^{[l+1]} dZ_j^{[l+1](i)}$$

Vectorizing across all hidden units and batch samples:

$$dA^{[l]} = \left(W^{[l+1]}\right)^T dZ^{[l+1]} \in \mathbb{R}^{n_l \times m}$$

$$dZ^{[l]} = dA^{[l]} \odot g'\left(Z^{[l]}\right) \in \mathbb{R}^{n_l \times m}$$

where $\odot$ is the element-wise Hadamard product and $g'(Z) = \mathbb{I}(Z > 0)$ for ReLU.

---

### 6. Summary of Generalized Recursion & SGD Update

For any layer $l \in \{1, \dots, L\}$:

$$dZ^{[L]} = A^{[L]} - Y \quad (\text{Output layer})$$

$$dA^{[l]} = \left(W^{[l+1]}\right)^T dZ^{[l+1]}$$

$$dZ^{[l]} = dA^{[l]} \odot \mathbb{I}\left(Z^{[l]} > 0\right) \quad (\text{Hidden layers } l < L)$$

$$dW^{[l]} = \frac{1}{m} dZ^{[l]} \left(A^{[l-1]}\right)^T$$

$$dB^{[l]} = \frac{1}{m} \sum_{i=1}^{m} dZ^{[l]}$$

**Stochastic Gradient Descent Parameter Update:**

$$W^{[l]} \leftarrow W^{[l]} - \alpha \, dW^{[l]}$$

$$B^{[l]} \leftarrow B^{[l]} - \alpha \, dB^{[l]}$$

---

## Codebase Structure

```text
neural-engine-numpy/
├── main.py                # MNIST dataset loader, pipeline setup & training loop
├── NEURAL ENGINE.pdf      # Detailed handwritten math derivations & notes
├── data.csv               # MNIST dataset CSV (60,000 samples)
├── LICENSE                # MIT License file
├── AGENTS.md              # Tensor shape invariants & design rules
└── nn/                    # Core Neural Engine Framework
    ├── __init__.py        # Package initialization & exports
    ├── init.py            # Alias module
    ├── layers.py          # Layer implementations (Dense, ReLU, SoftmaxCrossEntropy)
    ├── models.py          # Sequential model container & mini-batch trainer
    └── optimizers.py      # Optimization algorithms (SGD)
```

---

## Getting Started

### Prerequisites

Ensure Python 3.8+ and standard dependencies are installed:

```bash
pip install numpy pandas
```

### Running MNIST Training

Execute the main training script:

```bash
python main.py
```

### Defining Custom Architectures

```python
import numpy as np
from nn import Sequential, Dense, ReLU, SoftmaxCrossEntropy, SGD

# Construct neural network architecture
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

# Train model on X_train (784, m) and Y_train (10, m)
model.fit(X_train, Y_train, epochs=10, batch_size=64)

# Evaluate predictions
probs = model.predict(X_test)
predictions = np.argmax(probs, axis=0)
```

---

## MNIST Benchmark & Training Results

Training a `784 -> 128 -> 64 -> 10` architecture with batch size `64` and learning rate $\alpha = 0.1$:

| Epoch | Training Loss | Training Accuracy | Test Accuracy |
| :---: | :-----------: | :---------------: | :-----------: |
| **0** |  `0.354731`   |     `92.98%`      |       —       |
| **1** |  `0.168319`   |     `96.49%`      |       —       |
| **2** |  `0.120606`   |     `97.34%`      |       —       |
| **3** |  `0.094111`   |     `98.04%`      |       —       |
| **4** |  `0.076494`   |     `98.33%`      |       —       |
| **5** |  `0.064004`   |     `98.65%`      |       —       |
| **6** |  `0.053954`   |     `98.85%`      |       —       |
| **7** |  `0.045602`   |     `98.67%`      |       —       |
| **8** |  `0.039642`   |     `99.13%`      |       —       |
| **9** |  `0.033100`   |   **`99.43%`**    | **`97.29%`**  |

---

## Future Roadmap

The following modular extensions are planned for future development:

### Advanced Optimizers

- **SGD with Momentum**:
  $$V_{dW} = \beta V_{dW} + (1-\beta) dW, \qquad W \leftarrow W - \alpha V_{dW}$$
- **RMSProp**:
  $$S_{dW} = \beta S_{dW} + (1-\beta) dW^2, \qquad W \leftarrow W - \alpha \frac{dW}{\sqrt{S_{dW} + \epsilon}}$$
- **Adam (Adaptive Moment Estimation)**:
  $$W \leftarrow W - \alpha \frac{\hat{V}_{dW}}{\sqrt{\hat{S}_{dW}} + \epsilon}$$

### Regularization & Architectural Layers

- **Inverted Dropout**:
  Forward mask $M \sim \text{Bernoulli}(p)$, $A_{drop} = \frac{A \odot M}{p}$, backward $dA_{drop} = \frac{dA \odot M}{p}$.
- **Batch Normalization**:
  Normalizing mini-batch mean $\mu_B$ and variance $\sigma_B^2$ with learnable scale $\gamma$ and shift $\beta$:
  $$\hat{X} = \frac{X - \mu_B}{\sqrt{\sigma_B^2 + \epsilon}}, \qquad Y_{BN} = \gamma \hat{X} + \beta$$
- **$L_2$ Weight Decay (Ridge Regularization)**:
  Adding $\frac{\lambda}{2m} \Vert{}W\Vert{}^2$ penalty to cost and $\frac{\lambda}{m} W$ to weight gradients.

### Additional Activations & Loss Functions

- **LeakyReLU Activation**: $f(z) = \max(\alpha z, z)$ (resolves dying ReLU problem).
- **Tanh Activation**: $f(z) = \tanh(z)$, $f'(z) = 1 - \tanh^2(z)$.
- **Sigmoid & Binary Cross-Entropy**: For multi-label binary classification tasks.

### Verification & Diagnostic Utilities

- **Finite-Difference Numerical Gradient Checking (`gradcheck`)**:
  $$\frac{\partial \mathcal{L}}{\partial \theta} \approx \frac{\mathcal{L}(\theta + \epsilon) - \mathcal{L}(\theta - \epsilon)}{2\epsilon}$$

---

## References & Attribution

- **Video Tutorial**: [Explaining The Entire Math & Coding A Neural Engine From Scratch Using Only NumPy](https://youtu.be/KnZg2GKFDcQ) by **Aryan Mishra** ([@modestpenguinn](https://youtube.com/@modestpenguinn)).
- **Handwritten Mathematics**: Refer to [NEURAL ENGINE.pdf](NEURAL%20ENGINE.pdf) for the original derivations.

---

## License

Distributed under the **MIT License**. See `LICENSE` for details.
