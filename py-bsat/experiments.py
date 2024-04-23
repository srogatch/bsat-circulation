import numpy as np

A = np.array([
  [0.5, 0, 0],
  [0, 0.5, 0],
  [0, 0, -2]
])

eigenvalues = np.linalg.eigvals(A)
is_positive_semidefinite = np.all(eigenvalues >= 0)
print("Matrix A is positive semidefinite:", is_positive_semidefinite)
