# Deep Learning Lab 1
## Perceptron as the Smallest Decision-Making Model

### Tasks Completed
- Implemented perceptron from scratch (no ML libraries)
- Trained on AND, OR, NAND, NOR, XOR gates
- Printed final weights and bias
- Verified predictions

---

## Effect of Learning Rate

The learning rate (η) controls the magnitude of weight and bias updates during training.

- A small learning rate results in slow but stable learning.
- A large learning rate may cause instability or oscillation.
- It determines the trade-off between convergence speed and stability.

---

## Why the Same Code Learned Different Gates

The perceptron structure and learning rule remained unchanged for all logic gates.

Only the dataset was modified.

Since learning is data-driven, different input-output mappings resulted in different learned decision boundaries.

Therefore, behavior is defined by data, not architecture.

---

## Observation About XOR

The perceptron successfully learned AND, OR, NAND, and NOR gates.

However, it failed to learn XOR because XOR is not linearly separable.

A single perceptron can only learn linearly separable functions.