# Linear Regression Algorithm

## Overview

This repository contains a simple implementation of the linear regression algorithm from scratch using Python and NumPy. Linear regression is a fundamental machine learning algorithm used for predicting a continuous target variable based on one or more input features.

## Implementation Details

### LinearRegression Class

The core of the implementation is the `LinearRegression` class, which encapsulates the linear regression model. Here's a brief overview of its key methods:

#### `__init__(self, learning_rate=0.01, num_iterations=1000)`

- Initializes the linear regression model with default or user-defined learning rate and the number of iterations for gradient descent.

#### `fit(self, X, y)`

- Fits the model to the training data using gradient descent.
- `X`: Input features.
- `y`: Target variable.

#### `predict(self, X)`

- Predicts the target variable for new input features.
- `X`: Input features.

### Workflow

1. **Initialization:** Create an instance of the `LinearRegression` class, specifying hyperparameters if needed.

2. **Data Preparation:** Prepare your training data, ensuring it's in the appropriate format (NumPy arrays).

3. **Model Training:** Call the `fit` method to train the model on the training data.

4. **Prediction:** Use the `predict` method to make predictions on new data.


The optimization objective (mean squared error) is given by the formula:

\[ J(\theta) = \frac{1}{2m} \sum_{i=1}^{m} (h_\theta(x^{(i)}) - y^{(i)})^2 \]

where:
- \( J(\theta) \) is the cost function.
- \( m \) is the number of training examples.
- \( h_\theta(x) \) is the hypothesis function.
- \( x^{(i)} \) are the input features for the \( i \)-th example.
- \( y^{(i)} \) is the target variable for the \( i \)-th example.


## Usage

### Synthetic Data Example

```python
import numpy as np
import matplotlib.pyplot as plt
from LinearRegressionAlgo import LinearRegression

# Generate synthetic data for demonstration
np.random.seed(42)
X_synthetic = 2 * np.random.rand(100, 1)
y_synthetic = 4 + 3 * X_synthetic + np.random.randn(100, 1)

# Create and train the linear regression model
model = LinearRegression(learning_rate=0.01, num_iterations=1000)
model.fit(X_synthetic, y_synthetic)

# Make predictions
predictions_synthetic = model.predict(X_synthetic)

# Plot the synthetic data and the linear regression line
plt.scatter(X_synthetic, y_synthetic, label='Synthetic Data')
plt.plot(X_synthetic, predictions_synthetic, color='red', label='Linear Regression')
plt.xlabel('X')
plt.ylabel('y')
plt.title('Linear Regression with Synthetic Data')
plt.legend()
plt.show()
```

The linear regression model predicts the target variable \( y \) using the formula:

\[ y = \theta_0 + \theta_1 \cdot x \]
