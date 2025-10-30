
import numpy as np

class HebbNet:
    def __init__(self, n_inputs):
        self.weights = np.zeros(n_inputs)   
        self.bias = 0

    def train(self, X, Y):
        """
        X: input matrix shape (samples, features)
        Y: target outputs shape (samples,)
        """
        for x, y in zip(X, Y):
            self.weights += x * y
            self.bias += y

    def predict(self, x):
        # Activation = step function
        net = np.dot(self.weights, x) + self.bias
        return 1 if net >= 0 else -1

X = np.array([
    [-1, -1],
    [-1, +1],
    [+1, -1],
    [+1, +1]
])
Y_and = np.array([-1, -1, -1, +1])  # AND
Y_or  = np.array([-1, +1, +1, +1])  # OR

# Train AND
and_net = HebbNet(n_inputs=2)
and_net.train(X, Y_and)

print("AND function results:")
for x in X:
    print(f"{x} -> {and_net.predict(x)}")

# Train OR
or_net = HebbNet(n_inputs=2)
or_net.train(X, Y_or)

print("\nOR function results:")
for x in X:
    print(f"{x} -> {or_net.predict(x)}")