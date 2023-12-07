# PCA Class for Housing Data

## Overview

This Python script (`pca_class.py`) includes a PCA class that performs Principal Component Analysis (PCA) on the `housing.csv` dataset. The class is designed to be modular and reusable, allowing for easy integration into different projects.

## PCA Class Implementation

The `PCA` class includes the following methods:
- `__init__(self, n_components)`: Constructor to initialize the PCA object with the desired number of components.
- `fit_transform(self, X)`: Method to fit the PCA model and transform the input data.
- `inverse_transform(self, X_pca)`: Method to inverse transform PCA-reduced data back to the original space.

## Usage

1. Install required libraries:
   ```bash
   pip install numpy pandas matplotlib scikit-learn
