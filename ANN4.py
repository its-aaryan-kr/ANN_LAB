import numpy as np
import matplotlib.pyplot as plt

# Input range
x = np.linspace(-10, 10, 500)

# Activation functions
def step(x):
    return np.where(x >= 0, 1, 0)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def tanh(x):
    return np.tanh(x)

def relu(x):
    return np.maximum(0, x)

def leaky_relu(x, alpha=0.1):
    return np.where(x > 0, x, alpha * x)

def elu(x, alpha=1.0):
    return np.where(x > 0, x, alpha * (np.exp(x) - 1))

def swish(x):
    return x * sigmoid(x)

def softplus(x):
    return np.log(1 + np.exp(x))

# Dictionary of functions
functions = {
    "Step": step(x),
    "Sigmoid": sigmoid(x),
    "Tanh": tanh(x),
    "ReLU": relu(x),
    "Leaky ReLU": leaky_relu(x),
    "ELU": elu(x),
    "Swish": swish(x),
    "Softplus": softplus(x)
}

# Plotting
plt.figure(figsize=(14, 10))
for i, (name, y) in enumerate(functions.items(), 1):
    plt.subplot(3, 3, i)
    plt.plot(x, y, label=name, color='teal')
    plt.title(name, fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.axhline(0, color='black', linewidth=0.7)
    plt.axvline(0, color='black', linewidth=0.7)
    plt.legend()

plt.tight_layout()
plt.show()
