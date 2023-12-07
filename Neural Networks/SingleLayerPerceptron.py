import numpy as np
import matplotlib.pyplot as plt

class SingleLayerPerceptron:
    def __init__(self, input_size):
        self.weights = np.random.rand(input_size)
        self.bias = np.random.rand(1)

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def predict(self, X):
        z = np.dot(X, self.weights) + self.bias
        return self.sigmoid(z)

    def train(self, X, y, learning_rate=0.01, epochs=1000, epsilon=1e-8):
        for epoch in range(epochs):
            # Forward pass
            z = np.dot(X, self.weights) + self.bias
            predictions = self.sigmoid(z)

            # Compute the binary cross-entropy loss with epsilon for numerical stability
            loss = -np.mean(y * np.log(predictions + epsilon) + (1 - y) * np.log(1 - predictions + epsilon))

            # Backward pass
            dw = np.dot(X.T, predictions - y)
            db = np.sum(predictions - y)

            # Update weights and bias
            self.weights -= learning_rate * dw
            self.bias -= learning_rate * db

            # Visualize decision boundary every 100 epochs
            if epoch % 100 == 0:
                self.plot_decision_boundary(X, y, epoch)

    def plot_decision_boundary(self, X, y, epoch):
        plt.figure(figsize=(8, 6))
        plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.RdYlBu, edgecolors='k')
        plt.title(f'Decision Boundary - Epoch {epoch}')
        plt.xlabel('Feature 1')
        plt.ylabel('Feature 2')

        # Plot decision boundary
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1

        xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01),
                             np.arange(y_min, y_max, 0.01))

        Z = self.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)

        plt.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.RdYlBu)
        plt.show()


# Generate synthetic data for binary classification
np.random.seed(42)
X = np.random.randn(200, 2)
y = (X[:, 0] + X[:, 1] > 0).astype(int)

# Train the Single-Layer Perceptron
slp = SingleLayerPerceptron(input_size=2)
slp.train(X, y, learning_rate=0.1, epochs=1000)
