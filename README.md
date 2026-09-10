# 🧠 Neural Engine NumPy

<div align="center">

![Python](https://img.shields.io/badge/Python-3.8%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-Pure%20Matrix%20Math-013243?style=for-the-badge&logo=numpy&logoColor=white)
![Frameworks](https://img.shields.io/badge/Frameworks-Zero%20(Pure%20First%20Principles)-red?style=for-the-badge)
![License](https://img.shields.io/badge/License-MIT-blue?style=for-the-badge)

*A high-performance, modular deep learning framework built from first principles using pure Python and NumPy—no PyTorch, no TensorFlow, no Autograd.*

---

### 📹 Watch the Complete Math & Code Walkthrough
[![YouTube Video](https://img.youtube.com/vi/KnZg2GKFDcQ/maxresdefault.jpg)](https://youtu.be/KnZg2GKFDcQ)
*Click the banner above to watch the full step-by-step mathematical derivation and code walkthrough video.*

</div>

---

## 📌 Table of Contents

- [Overview](#-overview)
- [Key Features](#-key-features)
- [Architecture & Neural Pipeline](#-architecture--neural-pipeline)
- [Comprehensive Mathematical Foundations](#-comprehensive-mathematical-foundations)
  - [1. Matrix Notation & Tensor Shape Invariants](#1-matrix-notation--tensor-shape-invariants)
  - [2. He (Kaiming) Weight Initialization Derivation](#2-he-kaiming-weight-initialization-derivation)
  - [3. Forward Propagation Equations](#3-forward-propagation-equations)
  - [4. Categorical Cross-Entropy Loss](#4-categorical-cross-entropy-loss)
  - [5. Analytical Backpropagation Proof (Step-by-Step)](#5-analytical-backpropagation-proof-step-by-step)
    - [Step 5.1: Fused Softmax + Cross-Entropy Loss Gradient ($dZ^{[L]}$)](#step-51-fused-softmax--cross-entropy-loss-gradient-dzl)
    - [Step 5.2: Layer $L$ Parameter Gradients ($dW^{[L]}, dB^{[L]}$)](#step-52-layer-l-parameter-gradients-dwl-dbl)
    - [Step 5.3: Propagating Error to Hidden Layers ($dZ^{[l]}$)](#step-53-propagating-error-to-hidden-layers-dzl)
  - [6. Summary of Generalized Recursion & SGD Update](#6-summary-of-generalized-recursion--sgd-update)
- [Codebase Structure](#-codebase-structure)
- [Getting Started](#-getting-started)
- [MNIST Benchmark & Training Results](#-mnist-benchmark--training-results)
- [🔮 Future Roadmap (Planned Enhancements)](#-future-roadmap-planned-enhancements)
- [License](#-license)

---

## ⚡ Overview

**Neural Engine NumPy** is a deep learning engine built strictly from first principles using Python and NumPy. It eliminates black-box automatic differentiation engines (like PyTorch's `autograd`) in favor of explicit analytical matrix calculus.

Every tensor operation, weight initialization, activation function, loss computation, gradient propagation, and parameter update is derived on paper and translated directly into fully vectorized NumPy matrix operations (`np.dot` / `@`).

---

## ✨ Key Features

- 🚫 **Zero External ML Libraries**: Built exclusively with Python standard library and NumPy.
- 📐 **First-Principles Calculus**: Complete mathematical derivations using the multivariate chain rule and Softmax Jacobian matrix reductions.
- 🚀 **Fully Vectorized (Batch-First Matrix Algebra)**: Computes transformations across $m$ training instances in parallel without Python `for` loops over batch items.
- 🛡️ **Numerical Stability Guards**:
  - Max-logit subtraction in Softmax ($Z_{\text{shifted}} = Z - \max(Z)$) to prevent exponential overflow (`inf`).
  - Probability clipping ($\text{clip}(A, 10^{-15}, 1 - 10^{-15})$) to prevent $\ln(0)$ underflow (`NaN`).
  - He (Kaiming) normal weight initialization to maintain constant variance across deep ReLU layers.
- 🧱 **Clean Modular API**: Designed with object-oriented abstractions: `Dense`, `ReLU`, `SoftmaxCrossEntropy`, `SGD`, and `Sequential`.

---

## 🏗️ Architecture & Neural Pipeline

The baseline implementation classifies $28 \times 28$ pixel handwritten digits from the **MNIST** dataset into 10 classes (`0-9`):

```
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

## 🧮 Comprehensive Mathematical Foundations

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

### 2. He (Kaiming) Weight Initialization Derivation

Standard Xavier initialization assumes linear activations centered at 0 with variance $\text{Var}(W) = \frac{1}{n_{in}}$. However, ReLU sets all negative inputs to 0, zeroing out half of the variance:

$$\text{E}[\text{ReLU}(z)^2] = \frac{1}{2} \text{Var}(z)$$

For layer input $z = \sum_{k=1}^{n_{in}} w_k a_k$, under zero-mean independent variables:

$$\text{Var}(z) = n_{in} \cdot \text{Var}(w) \cdot \text{E}[a^2] = n_{in} \cdot \text{Var}(w) \cdot \left( \frac{1}{2} \text{Var}(z_{prev}) \right)$$

To enforce $\text{Var}(z) = \text{Var}(z_{prev}) = 1$ across arbitrarily deep networks:

$$\frac{1}{2} n_{in} \cdot \text{Var}(w) = 1 \implies \mathbf{\text{Var}(W) = \frac{2}{n_{in}}}$$

Thus, weights are initialized as:

$$W^{[l]} \sim \mathcal{N}\left(0, \, \sqrt{\frac{2}{n_{l-1}}}\right)$$

---

### 3. Forward Propagation Equations

For layer $l \in \{1, \dots, L\}$:

#### Affine Step:
$$Z^{[l]} = W^{[l]} A^{[l-1]} + B^{[l]}$$

#### Hidden Layer Activation (ReLU):
$$A^{[l]} = g(Z^{[l]}) = \max\left(0, \, Z^{[l]}\right)$$

#### Output Layer Activation (Softmax):
To avoid numerical overflow, subtract $\max(Z^{(i)})$ along the class axis:

$$\hat{z}_j^{[L](i)} = z_j^{[L](i)} - \max_{r} z_r^{[L](i)}$$

$$a_j^{[L](i)} = \frac{e^{\hat{z}_j^{[L](i)}}}{\sum_{r=1}^{n_L} e^{\hat{z}_r^{[L](i)}}}$$

---

### 4. Categorical Cross-Entropy Loss

For a single sample $(i)$ with one-hot label vector $y^{(i)}$:

$$\mathcal{L}^{(i)} = -\sum_{j=1}^{n_L} y_j^{(i)} \ln\left(a_j^{[L](i)}\right)$$

Average loss across mini-batch of size $m$:

$$\mathcal{L} = \frac{1}{m} \sum_{i=1}^{m} \mathcal{L}^{(i)} = -\frac{1}{m} \sum_{i=1}^{m} \sum_{j=1}^{n_L} y_j^{(i)} \ln\left(a_j^{[L](i)}\right)$$

---

### 5. Analytical Backpropagation Proof (Step-by-Step)

#### Step 5.1: Fused Softmax + Cross-Entropy Loss Gradient ($dZ^{[L]}$)

We compute $\frac{\partial \mathcal{L}^{(i)}}{\partial z_j^{[L](i)}}$. Since pre-activation $z_j^{[L]}$ influences the denominator of every output activation $a_k^{[L]}$, we apply the multivariable chain rule over all output neurons $k \in \{1, \dots, n_L\}$:

$$\frac{\partial \mathcal{L}^{(i)}}{\partial z_j^{[L](i)}} = \sum_{k=1}^{n_L} \frac{\partial \mathcal{L}^{(i)}}{\partial a_k^{[L](i)}} \frac{\partial a_k^{[L](i)}}{\partial z_j^{[L](i)}}$$

**1. Derivative of Loss w.r.t. Activation:**

$$\frac{\partial \mathcal{L}^{(i)}}{\partial a_k^{[L](i)}} = -\frac{y_k^{(i)}}{a_k^{[L](i)}}$$

**2. Softmax Jacobian Matrix (Quotient Rule):**
Using $a_k = \frac{e^{z_k}}{S}$ where $S = \sum_{r} e^{z_r}$:

- **Direct Path ($k = j$):**
  $$\frac{\partial a_j}{\partial z_j} = \frac{e^{z_j} S - e^{z_j} e^{z_j}}{S^2} = \frac{e^{z_j}}{S} \left(1 - \frac{e^{z_j}}{S}\right) = a_j (1 - a_j)$$

- **Indirect Path ($k \neq j$):**
  $$\frac{\partial a_k}{\partial z_j} = \frac{0 \cdot S - e^{z_k} e^{z_j}}{S^2} = -\left(\frac{e^{z_k}}{S}\right) \left(\frac{e^{z_j}}{S}\right) = -a_k a_j$$

**3. Expanding and Collapsing the Chain Rule Sum:**

$$\frac{\partial \mathcal{L}^{(i)}}{\partial z_j^{[L](i)}} = \left(-\frac{y_j^{(i)}}{a_j^{[L](i)}}\right) a_j^{[L](i)}(1 - a_j^{[L](i)}) + \sum_{k \neq j}^{n_L} \left(-\frac{y_k^{(i)}}{a_k^{[L](i)}}\right) \left(-a_k^{[L](i)} a_j^{[L](i)}\right)$$

$$= -y_j^{(i)} (1 - a_j^{[L](i)}) + \sum_{k \neq j}^{n_L} y_k^{(i)} a_j^{[L](i)}$$

$$= -y_j^{(i)} + y_j^{(i)} a_j^{[L](i)} + a_j^{[L](i)} \sum_{k \neq j}^{n_L} y_k^{(i)}$$

$$= -y_j^{(i)} + a_j^{[L](i)} \left( y_j^{(i)} + \sum_{k \neq j}^{n_L} y_k^{(i)} \right)$$

$$= -y_j^{(i)} + a_j^{[L](i)} \left( \sum_{k=1}^{n_L} y_k^{(i)} \right)$$

Because $Y$ is a one-hot distribution ($\sum_{k=1}^{n_L} y_k^{(i)} = 1$):

$$\frac{\partial \mathcal{L}^{(i)}}{\partial z_j^{[L](i)}} = a_j^{[L](i)} - y_j^{(i)}$$

Vectorized across all classes and $m$ batch samples:

$$\mathbf{dZ^{[L]} = A^{[L]} - Y} \in \mathbb{R}^{n_L \times m}$$

---

#### Step 5.2: Layer $L$ Parameter Gradients ($dW^{[L]}, dB^{[L]}$)

Applying the chain rule through $Z^{[L]} = W^{[L]} A^{[L-1]} + B^{[L]}$:

$$\frac{\partial \mathcal{L}^{(i)}}{\partial w_{jk}^{[L]}} = \frac{\partial \mathcal{L}^{(i)}}{\partial z_j^{[L](i)}} \frac{\partial z_j^{[L](i)}}{\partial w_{jk}^{[L]}} = dZ_j^{[L](i)} a_k^{[L-1](i)}$$

Averaging across mini-batch size $m$:

$$\mathbf{dW^{[L]} = \frac{1}{m} dZ^{[L]} \left(A^{[L-1]}\right)^T} \in \mathbb{R}^{n_L \times n_{L-1}}$$

$$\mathbf{dB^{[L]} = \frac{1}{m} \sum_{i=1}^{m} dZ^{[L](i)} = \frac{1}{m} \text{np.sum}(dZ^{[L]}, \text{axis}=1, \text{keepdims}=\text{True})} \in \mathbb{R}^{n_L \times 1}$$

---

#### Step 5.3: Propagating Error to Hidden Layers ($dZ^{[l]}$)

To push gradient back from layer $l+1$ to layer $l$:

$$\frac{\partial \mathcal{L}^{(i)}}{\partial z_k^{[l](i)}} = \sum_{j=1}^{n_{l+1}} \frac{\partial \mathcal{L}^{(i)}}{\partial z_j^{[l+1](i)}} \frac{\partial z_j^{[l+1](i)}}{\partial a_k^{[l](i)}} \frac{\partial a_k^{[l](i)}}{\partial z_k^{[l](i)}}$$

Using:
- $\frac{\partial \mathcal{L}^{(i)}}{\partial z_j^{[l+1](i)}} = dZ_j^{[l+1](i)}$
- $\frac{\partial z_j^{[l+1](i)}}{\partial a_k^{[l](i)}} = W_{jk}^{[l+1]}$
- $\frac{\partial a_k^{[l](i)}}{\partial z_k^{[l](i)}} = g'(z_k^{[l](i)})$

$$\frac{\partial \mathcal{L}^{(i)}}{\partial z_k^{[l](i)}} = g'\left(z_k^{[l](i)}\right) \sum_{j=1}^{n_{l+1}} W_{jk}^{[l+1]} dZ_j^{[l+1](i)}$$

Vectorizing across all hidden units and batch samples:

$$\mathbf{dA^{[l]} = \left(W^{[l+1]}\right)^T dZ^{[l+1]}} \in \mathbb{R}^{n_l \times m}$$

$$\mathbf{dZ^{[l]} = dA^{[l]} \odot g'\left(Z^{[l]}\right)} \in \mathbb{R}^{n_l \times m}$$

where $\odot$ is the element-wise Hadamard product and $g'(Z) = \mathbb{I}(Z > 0)$ for ReLU.

---

### 6. Summary of Generalized Recursion & SGD Update

For any layer $l \in \{1, \dots, L\}$:

$$\begin{aligned}
\mathbf{dZ^{[L]}} &= A^{[L]} - Y \quad (\text{Output layer}) \\
\mathbf{dA^{[l]}} &= \left(W^{[l+1]}\right)^T dZ^{[l+1]} \\
\mathbf{dZ^{[l]}} &= dA^{[l]} \odot \mathbb{I}(Z^{[l]} > 0) \quad (\text{Hidden layers } l < L) \\
\mathbf{dW^{[l]}} &= \frac{1}{m} dZ^{[l]} \left(A^{[l-1]}\right)^T \\
\mathbf{dB^{[l]}} &= \frac{1}{m} \sum_{i=1}^{m} dZ^{[l]}
\end{aligned}$$

**Stochastic Gradient Descent Parameter Update:**

$$\mathbf{W^{[l]} \leftarrow W^{[l]} - \alpha \, dW^{[l]}}$$

$$\mathbf{B^{[l]} \leftarrow B^{[l]} - \alpha \, dB^{[l]}}$$

---

## 📁 Codebase Structure

```
neural-engine-numpy/
├── main.py                # MNIST dataset loader, pipeline setup & training loop
├── NEURAL ENGINE.pdf      # Detailed handwritten math derivations & notes
├── data.csv               # MNIST dataset CSV (60,000 samples)
├── AGENTS.md              # Tensor shape invariants & design rules
└── nn/                    # Core Neural Engine Framework
    ├── __init__.py        # Package initialization & exports
    ├── init.py            # Alias module
    ├── layers.py          # Layer implementations (Dense, ReLU, SoftmaxCrossEntropy)
    ├── models.py          # Sequential model container & mini-batch trainer
    └── optimizers.py      # Optimization algorithms (SGD)
```

---

## 🚀 Getting Started

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

## 📊 MNIST Benchmark & Training Results

Training a `784 -> 128 -> 64 -> 10` architecture with batch size `64` and learning rate $\alpha = 0.1$:

| Epoch | Training Loss | Training Accuracy | Test Accuracy |
| :---: | :---: | :---: | :---: |
| **0** | `0.354731` | `92.98%` | — |
| **1** | `0.168319` | `96.49%` | — |
| **2** | `0.120606` | `97.34%` | — |
| **3** | `0.094111` | `98.04%` | — |
| **4** | `0.076494` | `98.33%` | — |
| **5** | `0.064004` | `98.65%` | — |
| **6** | `0.053954` | `98.85%` | — |
| **7** | `0.045602` | `98.67%` | — |
| **8** | `0.039642` | `99.13%` | — |
| **9** | `0.033100` | **`99.43%`** | **`97.29%`** |

---

## 🔮 Future Roadmap (Planned Enhancements)

The following modular extensions are planned for future development to expand framework capabilities:

### ⚡ Advanced Optimizers
- [ ] **SGD with Momentum**:
  $$V_{dW} = \beta V_{dW} + (1-\beta) dW, \qquad W \leftarrow W - \alpha V_{dW}$$
- [ ] **RMSProp**:
  $$S_{dW} = \beta S_{dW} + (1-\beta) dW^2, \qquad W \leftarrow W - \alpha \frac{dW}{\sqrt{S_{dW} + \epsilon}}$$
- [ ] **Adam (Adaptive Moment Estimation)**:
  First ($V$) and second ($S$) moment estimates with bias corrections ($\hat{V}, \hat{S}$):
  $$W \leftarrow W - \alpha \frac{\hat{V}_{dW}}{\sqrt{\hat{S}_{dW}} + \epsilon}$$

### 🛡️ Regularization & Architectural Layers
- [ ] **Inverted Dropout**:
  Forward mask $M \sim \text{Bernoulli}(p)$, $A_{drop} = \frac{A \odot M}{p}$, backward $dA_{drop} = \frac{dA \odot M}{p}$.
- [ ] **Batch Normalization**:
  Normalizing mini-batch mean $\mu_B$ and variance $\sigma_B^2$ with learnable scale $\gamma$ and shift $\beta$:
  $$\hat{X} = \frac{X - \mu_B}{\sqrt{\sigma_B^2 + \epsilon}}, \qquad Y_{BN} = \gamma \hat{X} + \beta$$
- [ ] **$L_2$ Weight Decay (Ridge Regularization)**:
  Adding $\frac{\lambda}{2m} \|W\|^2$ penalty to cost and $\frac{\lambda}{m} W$ to weight gradients.

### 🧪 Additional Activations & Loss Functions
- [ ] **LeakyReLU Activation**: $f(z) = \max(\alpha z, z)$ (resolves dying ReLU problem).
- [ ] **Tanh Activation**: $f(z) = \tanh(z)$, $f'(z) = 1 - \tanh^2(z)$.
- [ ] **Sigmoid & Binary Cross-Entropy**: For multi-label binary classification tasks.

### 🔍 Verification & Diagnostic Utilities
- [ ] **Finite-Difference Numerical Gradient Checking (`gradcheck`)**:
  Comparing analytical gradients against numerical approximation:
  $$\frac{\partial \mathcal{L}}{\partial \theta} \approx \frac{\mathcal{L}(\theta + \epsilon) - \mathcal{L}(\theta - \epsilon)}{2\epsilon}$$
  Ensuring relative error $\frac{\|\theta_{num} - \theta_{analytical}\|_2}{\|\theta_{num}\|_2 + \|\theta_{analytical}\|_2} < 10^{-7}$.

---

## 🤝 References & Attribution

- **Video Tutorial**: [Explaining The Entire Math & Coding A Neural Engine From Scratch Using Only NumPy](https://youtu.be/KnZg2GKFDcQ) by **Aryan Mishra** ([@modestpenguinn](https://youtube.com/@modestpenguinn)).
- **Handwritten Mathematics**: Refer to [NEURAL ENGINE.pdf](file:///home/aryanm/dev/neural-engine-numpy/NEURAL%20ENGINE.pdf) for the original derivations.

---

## 📜 License

Distributed under the MIT License. See `LICENSE` for details.
