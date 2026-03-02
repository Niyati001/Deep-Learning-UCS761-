# STEP1: Define activation function
import numpy as np

def relu(z):
    return np.maximum(0, z)

def relu_derivative(z):
    return (z > 0).astype(float)

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def sigmoid_derivative(z):
    s = sigmoid(z)
    return s * (1 - s)

def tanh(z):
    return np.tanh(z)

def tanh_derivative(z):
    return 1 - np.tanh(z)**2

def leaky_relu(z, alpha=0.01):
    return np.where(z > 0, z, alpha * z)

def leaky_relu_derivative(z, alpha=0.01):
    return np.where(z > 0, 1, alpha)

def softplus(z):
    return np.log(1 + np.exp(z))

def softplus_derivative(z):
    return sigmoid(z)

#STEP2: Start with Dataset
import numpy as np
np.random.seed(42)

X= np.random.uniform(-2,2,(400,3))
y=( np.sin(X[:, 0]) + 0.5* (X[:, 1]**2) - 0.8* X[:,2])
y= y.reshape(-1,1)

#STEP4: Forward pass
def forward(X, weights, biases, activation_func):
    activations = [X]
    Zs = []
    A = X

    for i in range(len(weights) - 1):
        Z = A @ weights[i] + biases[i]
        A = activation_func(Z)

        Zs.append(Z)
        activations.append(A)

    # Final layer (linear output)
    Z = A @ weights[-1] + biases[-1]
    Zs.append(Z)
    activations.append(Z)

    return activations, Zs


#STEP5: Backward
def backward(y, activations, Zs, weights, activation_derivative):
    grads_W = []
    grads_b = []
    m = y.shape[0]
    dA = 2 * (activations[-1] - y) / m

    for i in reversed(range(len(weights))):
        A_prev = activations[i]

        if i == len(weights) - 1:
            dZ = dA
        else:
            dZ = dA * activation_derivative(Zs[i])

        dW = A_prev.T @ dZ
        db = np.sum(dZ, axis=0, keepdims=True)

        grads_W.insert(0, dW)
        grads_b.insert(0, db)
        dA = dZ @ weights[i].T

    return grads_W, grads_b

#STEP6: Gradient Norm Measurement
def gradient_norm(grad):
    return np.sqrt(np.sum(grad ** 2))

#STEP7: Training Function
def initialize_network(layers):
    weights = []
    biases = []

    for i in range(len(layers) - 1):
        W = np.random.uniform(-0.5, 0.5, (layers[i], layers[i+1]))
        b = np.zeros((1, layers[i+1]))

        weights.append(W)
        biases.append(b)

    return weights, biases

def train_model(layers, activation_func, activation_derivative):
    weights, biases = initialize_network(layers)

    lr = 0.01
    epochs = 1000

    # Initialize gradient norms BEFORE loop
    grad_norm_first = None
    grad_norm_last = None

    for epoch in range(epochs):
        activations, Zs = forward(X, weights, biases, activation_func)
        loss = np.mean((activations[-1] - y) ** 2)

        grads_W, grads_b = backward(
            y, activations, Zs, weights, activation_derivative
        )

        if epoch == epochs - 1:
            grad_norm_first = gradient_norm(grads_W[0])
            grad_norm_last = gradient_norm(grads_W[-2])

        for i in range(len(weights)):
            weights[i] -= lr * grads_W[i]
            biases[i] -= lr * grads_b[i]

    return loss, grad_norm_first, grad_norm_last

#STEP8: All 4 architectures
models = {
    "Model A": [3, 4, 1],
    "Model B": [3, 6, 6, 1],
    "Model C": [3, 8, 8, 8, 8, 1],
    "Model D": [3, 8, 8, 8, 8, 8, 8, 8, 8, 1]
}

for name, layers in models.items():
    print("Running:", name, "with ReLU")
    loss, g1, gL = train_model(layers, relu, relu_derivative)
    print("Final Loss:", loss)
    print("Grad Norm First:", g1)
    print("Grad Norm Last Hidden:", gL)

    print("Running:", name, "with Sigmoid")
    loss, g1, gL = train_model(layers, sigmoid, sigmoid_derivative)
    print("Final Loss:", loss)
    print("Grad Norm First:", g1)
    print("Grad Norm Last Hidden:", gL)

    print("-" * 40)