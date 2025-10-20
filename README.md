# Neural-Network-From-Sratch
# 🧠 Single Layer Neural Network (from Scratch)

A simple **custom neural network** built entirely from **NumPy**, trained on the **Iris dataset** for multi-class classification.  
This project demonstrates how neural networks can be implemented from first principles — without using deep learning libraries like TensorFlow or PyTorch.

---

## 🚀 Features
- Implemented forward and backward propagation manually.
- Custom **SVM-style loss function** for multi-class classification.
- Gradient descent optimization with configurable learning rate and epochs.
- Built-in evaluation and accuracy calculation.
- Tested on the **Iris dataset** using `scikit-learn`.

---

## 🧩 Model Architecture
- **Input Layer:** Number of input features = 4 (for Iris dataset)
- **Output Layer:** 3 neurons (for each class)
- **Activation:** Linear (no hidden layers)
- **Loss Function:** Multi-class SVM Loss
- **Optimizer:** Manual Gradient Descent

---

## 🧠 Code Overview

```python
class SingleLayerNN:
    def __init__(self, input_size, output_size, lr=0.01, epochs=100):
        self.weights = np.random.randn(input_size, output_size)
        self.bias = np.zeros((1, output_size))
        self.lr = lr
        self.epochs = epochs

    def svm_loss(self, logits, y):
        # Computes hinge loss and its gradient
        ...

    def affine_forward(self, X):
        # Forward pass (linear transformation)
        return np.dot(X, self.weights) + self.bias

    def affine_backward(self, gradient, X):
        # Backward pass (compute weight and bias gradients)
        ...

    def train(self, X, y):
        # Training loop with gradient updates
        ...
