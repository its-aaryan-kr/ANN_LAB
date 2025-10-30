import numpy as np

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def dsigmoid(y):
    return y * (1 - y)

class SingleLayerNN:
    def __init__(self, input_size, output_size=1, lr=0.1, seed=0):
        np.random.seed(seed)
        self.lr = lr
        self.W = np.random.randn(output_size, input_size) * 0.01
        self.b = np.zeros((output_size, 1))

    def forward(self, X):
        z = np.dot(self.W, X) + self.b
        return sigmoid(z)

    def compute_loss(self, y_pred, y_true):
        m = y_true.shape[1]
        return 0.5 * np.sum((y_pred - y_true) ** 2) / m

    def train(self, X, Y, epochs=1000, print_every=100):
        m = Y.shape[1]
        for epoch in range(1, epochs + 1):
            # Forward
            y_pred = self.forward(X)
            # Compute error
            error = y_pred - Y
            # Gradient
            dW = np.dot(error * dsigmoid(y_pred), X.T) / m
            db = np.sum(error * dsigmoid(y_pred), axis=1, keepdims=True) / m
            self.W -= self.lr * dW
            self.b -= self.lr * db

            if epoch % print_every == 0 or epoch == 1:
                loss = self.compute_loss(y_pred, Y)
                print(f"Epoch {epoch:6d}  Loss: {loss:.6f}")

    def predict(self, X):
        y_pred = self.forward(X)
        return (y_pred > 0.5).astype(int)

def main():
    X = np.array([[0, 0, 1, 1],
                  [0, 1, 0, 1]])  
    Y = np.array([[0, 0, 0, 1]])   

    net = SingleLayerNN(input_size=2, output_size=1, lr=0.5, seed=42)
    net.train(X, Y, epochs=1000, print_every=200)

    print("\nPredictions:")
    print(net.predict(X))
    print("Expected:", Y)

if __name__ == "__main__":
    main()
