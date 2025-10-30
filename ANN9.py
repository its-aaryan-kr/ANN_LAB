import math
import matplotlib.pyplot as plt

class LVQ:
    def winner(self, weights, sample):
        """Find the winning prototype (closest weight vector)."""
        distances = []
        for w in weights:
            d = sum((sample[i] - w[i]) ** 2 for i in range(len(sample)))
            distances.append(math.sqrt(d))
        return distances.index(min(distances))

    def update(self, weights, sample, label, J, cluster_labels, alpha):
        """Update the weights based on whether the winner's label matches the sample's label."""
        if cluster_labels[J] == label:
            # Move winner closer
            for i in range(len(weights[J])):
                weights[J][i] += alpha * (sample[i] - weights[J][i])
        else:
            # Move winner away
            for i in range(len(weights[J])):
                weights[J][i] -= alpha * (sample[i] - weights[J][i])
        return weights


def format_weights(weights):
    """Format weights with 8 decimal places"""
    return [[round(v, 8) for v in w] for w in weights]


def main():
    # Initial weights (prototypes)
    weights = [
        [0.2, 0.6, 0.6, 0.3],
        [0.5, 0.6, 0.3, 0.3]
    ]

    # Training samples (features)
    T = [
        [0, 1, 0, 1],
        [0, 1, 1, 1],
        [0, 0, 0, 0],
        [1, 0, 0, 1]
    ]

    # True class labels
    labels = [0, 0, 1, 1]

    # Assign class labels to prototypes
    cluster_labels = [0, 1]

    # Test sample
    test_sample = [1, 0, 0, 1]

    lvq = LVQ()
    alpha = 0.3
    epochs = 10

    # ---- Training ----
    for epoch in range(epochs):
        print(f"\nEpoch {epoch + 1}:")
        for i, sample in enumerate(T):
            J = lvq.winner(weights, sample)
            weights = lvq.update(weights, sample, labels[i], J, cluster_labels, alpha)
            print(f"Sample {sample} -> Winner: Cluster {J}, Updated weights: {format_weights(weights)}")

        alpha *= 0.9  # decay learning rate

    print("\nFinal Result:")
    J_test = lvq.winner(weights, test_sample)
    print(f"Test Sample {test_sample} belongs to Cluster: {cluster_labels[J_test]}")
    print(f"Trained weights: {format_weights(weights)}")

    # ---- Visualization ----
    plt.figure(figsize=(6, 6))

    # Plot training samples (only first 2 features for visualization)
    for i, sample in enumerate(T):
        color = 'blue' if labels[i] == 0 else 'green'
        plt.scatter(sample[0], sample[1], c=color, marker='o', s=50)

    # Plot prototypes/clusters
    cluster_colors = ['red', 'purple']
    for j, w in enumerate(weights):
        plt.scatter(w[0], w[1], c=cluster_colors[j], marker='X', s=150, label=f'Cluster {cluster_labels[j]}')

    # Plot test sample
    plt.scatter(test_sample[0], test_sample[1], c='orange', marker='o', s=100, label='Test sample')

    plt.title("LVQ Clustering (2 clusters)")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.legend()
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    main()
