import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.preprocessing import MinMaxScaler

# Load and normalize the Iris dataset
iris = load_iris()
data = iris.data
labels = iris.target
scaler = MinMaxScaler()
data = scaler.fit_transform(data)

# SOM parameters
m, n, dim, iters = 5, 5, data.shape[1], 1000
W = np.random.rand(m, n, dim)
sigma0, lr0 = max(m, n) / 2, 0.3

# SOM training
for t in range(iters):
    x = data[np.random.randint(0, len(data))]
    bmu = np.unravel_index(np.argmin(np.linalg.norm(W - x, axis=2)), (m, n))
    lr = lr0 * np.exp(-t / iters)
    sigma = sigma0 * np.exp(-t / (iters / np.log(sigma0)))

    for i in range(m):
        for j in range(n):
            d = np.linalg.norm(np.array([i, j]) - bmu)
            h = np.exp(-d*2 / (2 * sigma*2))
            W[i, j] += lr * h * (x - W[i, j])

    if t % 200 == 0:
        print(f"Iteration {t}/{iters}")

# Mapping each input to its BMU (for cluster visualization)
mapped = np.zeros((len(data), 2))
for idx, x in enumerate(data):
    mapped[idx] = np.unravel_index(np.argmin(np.linalg.norm(W - x, axis=2)), (m, n))

# Plot SOM clusters
plt.figure(figsize=(8,6))
for i, label in enumerate(np.unique(labels)):
    plt.scatter(mapped[labels == label, 1], mapped[labels == label, 0], label=iris.target_names[label], s=50)

plt.title("Self Organizing Map (SOM) on Iris Dataset")
plt.gca().invert_yaxis()
plt.legend()
plt.show()