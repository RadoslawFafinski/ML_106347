import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.datasets import load_iris

iris = load_iris()

X = iris.data
y = iris.target

print("Kształt danych:", X.shape)
print("Nazwy cech:", iris.feature_names)

# 1. Średnie
mean_vector = np.mean(X, axis=0)

# 2. Centrowanie
X_centered = X - mean_vector

# 3. Macierz kowariancji
n = X_centered.shape[0]

covariance_matrix = (
    X_centered.T @ X_centered
) / (n - 1)

# 4. Wartości i wektory własne
eigenvalues, eigenvectors = np.linalg.eigh(
    covariance_matrix
)

# 5. Sortowanie malejąco
order = np.argsort(eigenvalues)[::-1]

eigenvalues = eigenvalues[order]
eigenvectors = eigenvectors[:, order]

print("Wartości własne:")
print(eigenvalues)

print("\nWektory własne:")
print(eigenvectors)