# Neural Engine NumPy

![Python](https://img.shields.io/badge/Python-3.8%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white)

A deep learning framework built entirely from first principles using pure Python and NumPy. No black-box autograd, no PyTorch, no TensorFlow.

## Resources

- **Walkthrough:** [YouTube Video](https://youtu.be/KnZg2GKFDcQ)
- **Calculus Derivations:** Reference `NEURAL ENGINE.pdf` in this repository for the full backpropagation and optimization proofs.

## Architecture

MNIST baseline model:
`Input (784) -> Dense(128) -> ReLU -> Dense(64) -> ReLU -> Dense(10) -> Softmax Cross-Entropy`

## Quick Start

**Dependencies:**

```bash
pip install numpy pandas
```

**Run Training:**

```bash
python main.py
```

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

## License

MIT
