# ffnn_numpy_xor.py
import numpy as np

# -----------------------
# Activation functions + derivatives
# -----------------------
def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def dsigmoid(y):
    # derivative w.r.t. output y = sigmoid(x) -> y*(1-y)
    return y * (1.0 - y)

# -----------------------
# Simple Feed-Forward Neural Network (dense MLP)
# -----------------------
class SimpleFFNN:
    def __init__(self, layer_sizes, seed=1):
        """
        layer_sizes: list, e.g. [2, 4, 1] -> input 2, one hidden layer 4, output 1
        """
        np.random.seed(seed)
        self.layer_sizes = layer_sizes
        self.num_layers = len(layer_sizes)
        # Initialize weights and biases: small random values
        self.weights = []
        self.biases = []
        for i in range(self.num_layers - 1):
            in_size = layer_sizes[i]
            out_size = layer_sizes[i + 1]
            # Xavier/Glorot-like initialization
            limit = np.sqrt(6.0 / (in_size + out_size))
            W = np.random.uniform(-limit, limit, (out_size, in_size))
            b = np.zeros((out_size, 1))
            self.weights.append(W)
            self.biases.append(b)

    def forward(self, x):
       
        a = x
        activations = [a]
        zs = []
        for W, b in zip(self.weights, self.biases):
            z = np.dot(W, a) + b
            a = sigmoid(z)
            zs.append(z)
            activations.append(a)
        return activations, zs

    def compute_loss(self, y_pred, y_true):
        """
        Mean Squared Error loss (for XOR small toy problem).
        y_pred, y_true shape: (1, batch)
        """
        m = y_true.shape[1]
        loss = 0.5 * np.sum((y_pred - y_true) ** 2) / m
        return loss

    def backprop(self, x, y):
        """
        Compute gradients for a single batch (x and y may be batches).
        Returns gradients lists for weights and biases with same shapes.
        """
        m = y.shape[1]  # batch size
        activations, zs = self.forward(x)

        # Initialize gradient containers
        nabla_w = [np.zeros_like(W) for W in self.weights]
        nabla_b = [np.zeros_like(b) for b in self.biases]

        # Compute delta for output layer: (a_L - y) * sigmoid'(z_L)
        delta = (activations[-1] - y) * dsigmoid(activations[-1])
        nabla_b[-1] = np.sum(delta, axis=1, keepdims=True) / m
        nabla_w[-1] = (np.dot(delta, activations[-2].T)) / m

        # Backpropagate through hidden layers
        for l in range(2, self.num_layers):
            z = zs[-l]
            sp = dsigmoid(activations[-l])  # derivative from activation output
            delta = np.dot(self.weights[-l + 1].T, delta) * sp
            nabla_b[-l] = np.sum(delta, axis=1, keepdims=True) / m
            nabla_w[-l] = (np.dot(delta, activations[-l - 1].T)) / m

        return nabla_w, nabla_b

    def update_parameters(self, nabla_w, nabla_b, lr):
        for i in range(len(self.weights)):
            self.weights[i] -= lr * nabla_w[i]
            self.biases[i]  -= lr * nabla_b[i]

    def train(self, X, Y, epochs=10000, lr=1.0, print_every=1000):
        """
        X shape: (n_input, m)
        Y shape: (n_output, m)
        """
        for epoch in range(1, epochs + 1):
            nabla_w, nabla_b = self.backprop(X, Y)
            self.update_parameters(nabla_w, nabla_b, lr)

            if epoch % print_every == 0 or epoch == 1:
                a_last, _ = self.forward(X)
                loss = self.compute_loss(a_last[-1], Y)
                print(f"Epoch {epoch:6d}  Loss: {loss:.6f}")

    def predict(self, X):
        activations, _ = self.forward(X)
        return activations[-1]

# -----------------------
# XOR dataset & training run
# -----------------------
def main():
    # XOR inputs (4 examples)
    X = np.array([[0, 0, 1, 1],
                  [0, 1, 0, 1]], dtype=float)  # shape (2,4)
    Y = np.array([[0, 1, 1, 0]], dtype=float)  # shape (1,4)

    # Convert to column-batched form (already in that shape: features x batch)
    # Build network: 2 inputs -> 4 hidden -> 1 output
    net = SimpleFFNN([2, 4, 1], seed=42)

    # Train
    net.train(X, Y, epochs=10000, lr=1.5, print_every=1000)

    # Predictions
    preds = net.predict(X)
    print("\nPredictions (raw outputs):")
    print(np.round(preds, 4))
    print("\nRounded to 0/1:")
    print((preds > 0.5).astype(int))

    # Show final weights (optional)
    # for i, W in enumerate(net.weights):
    #     print(f"W{i} =\n{W}\n b{i} =\n{net.biases[i]}")

if __name__ == "__main__":

    main()