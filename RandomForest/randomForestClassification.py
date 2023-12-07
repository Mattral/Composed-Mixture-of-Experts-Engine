import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.tree import plot_tree
from sklearn.metrics import accuracy_score

class DecisionTreeClassifier:
    def __init__(self, max_depth=None):
        self.max_depth = max_depth
        self.tree = None
        self.num_classes = None
        self.feature_importances_ = np.zeros(1)

    def fit(self, X, y):
        self.num_classes = len(np.unique(y))
        self.tree, self.feature_importances_ = self._grow_tree(X, y)

    def _gini(self, y):
        m = len(y)
        return 1.0 - sum((np.sum(y == c) / m) ** 2 for c in np.unique(y))

    def _best_split(self, X, y):
        m = len(y)
        if m <= 1:
            return None, None

        num_parent = [np.sum(y == c) for c in range(self.num_classes)]

        best_gini = 1.0 - sum((num / m) ** 2 for num in num_parent)
        best_idx, best_thr = None, None

        for idx in range(X.shape[1]):
            thresholds, classes = zip(*sorted(zip(X[:, idx], y)))

            num_left = [0] * self.num_classes
            num_right = num_parent.copy()

            for i in range(1, m):
                c = classes[i - 1]
                num_left[c] += 1
                num_right[c] -= 1
                gini_left = 1.0 - sum((num_left[x] / i) ** 2 for x in range(self.num_classes))
                gini_right = 1.0 - sum((num_right[x] / (m - i)) ** 2 for x in range(self.num_classes))

                gini = (i * gini_left + (m - i) * gini_right) / m

                if thresholds[i] == thresholds[i - 1]:
                    continue

                if gini < best_gini:
                    best_gini = gini
                    best_idx = idx
                    best_thr = (thresholds[i] + thresholds[i - 1]) / 2

        return best_idx, best_thr

    def _grow_tree(self, X, y, depth=0):

        num_samples_per_class = [np.sum(y == i) for i in np.unique(y)]
        predicted_class = np.argmax(num_samples_per_class)
        node = {'class': predicted_class, 'num_samples': len(y)}

        feature_importances = np.zeros(X.shape[1])

        if self.max_depth is None or depth < self.max_depth:
            idx, thr = self._best_split(X, y)
            if idx is not None:
                indices_left = X[:, idx] < thr
                X_left, y_left = X[indices_left], y[indices_left]
                X_right, y_right = X[~indices_left], y[~indices_left]
                node['index'] = idx
                node['threshold'] = thr
                left_node, left_importance = self._grow_tree(X_left, y_left, depth + 1)
                right_node, right_importance = self._grow_tree(X_right, y_right, depth + 1)
                node['left'] = left_node
                node['right'] = right_node

                # Update feature importances correctly
                feature_importances += left_importance
                feature_importances += right_importance

        return node, feature_importances

        
    def predict(self, X):
        return [self._predict(inputs) for inputs in X]

    def _predict(self, inputs):
        node = self.tree
        while 'index' in node:
            if inputs[node['index']] < node['threshold']:
                node = node['left']
            else:
                node = node['right']
        return node['class']

    def plot_tree(self, feature_names=None, class_names=None, filled=True):
        plt.figure(figsize=(10, 10))
        plt.xlabel(feature_names)
        plt.ylabel(class_names)
        self._plot_tree_rec(self.tree, feature_names, class_names, filled)
        plt.show()

    def _plot_tree_rec(self, node, feature_names, class_names, filled=True, indent=0):
        if node is None:
            return

        if 'left' not in node and 'right' not in node:
            label = f"{class_names[node['class']]} (class {node['class']})"
            print(f"{' ' * indent}Leaf: {label}")
            return

        print(f"{' ' * indent}Feature {feature_names[node['index']]} <= {node['threshold']:.2f}")

        print(f"{' ' * indent}Left:")
        self._plot_tree_rec(node['left'], feature_names, class_names, filled, indent + 2)

        print(f"{' ' * indent}Right:")
        self._plot_tree_rec(node['right'], feature_names, class_names, filled, indent + 2)


class RandomForestClassifier:
    def __init__(self, n_trees=100, max_depth=None):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.trees = []

    def fit(self, X, y):
        for i in range(self.n_trees):
            tree = DecisionTreeClassifier(max_depth=self.max_depth)
            indices = np.random.choice(len(X), len(X), replace=True)
            tree.fit(X[indices], y[indices])
            self.trees.append(tree)

            # Print information about the current iteration
            print(f"Iteration {i + 1} - Tree Depth: {tree.max_depth}")

    def predict(self, X):
        predictions = np.array([tree.predict(X) for tree in self.trees])
        return np.mean(predictions, axis=0).astype(int)


# Load the Iris dataset
iris = load_iris()
X = iris.data
y = iris.target

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the Decision Tree Classifier from scratch
dt_classifier = DecisionTreeClassifier(max_depth=None)
dt_classifier.fit(X_train, y_train)

# Access feature importance from the tree
feature_importance = dt_classifier.feature_importances_

# Print feature importance
for i, importance in enumerate(feature_importance):
    print(f"Feature {i}: {importance}")

# Create and train the Random Forest Classifier from scratch
rf_classifier = RandomForestClassifier(n_trees=100, max_depth=3)
rf_classifier.fit(X_train, y_train)

# Predict labels for the test set
y_pred = rf_classifier.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Final Accuracy: {accuracy}")

# Access feature importance from the first tree
feature_importance = rf_classifier.trees[0].feature_importances_

# Print feature importance
for i, importance in enumerate(feature_importance):
    print(f"Feature {i}: {importance}")

# Visualize the classified groups
plt.figure(figsize=(10, 6))
plt.scatter(X_test[:, 0], X_test[:, 1], c=y_pred, cmap='viridis', edgecolors='k', s=50, label='Predicted')
plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap='viridis', marker='X', edgecolors='r', s=100, label='True')
plt.title('Random Forest Classification - Predicted vs True')
plt.xlabel('Feature 0')
plt.ylabel('Feature 1')
plt.legend()
plt.show()
