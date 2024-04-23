import numpy as np

# A = np.array([
#   [0.5, 0, 0],
#   [0, 0.5, 0],
#   [0, 0, -2]
# ])

A = np.array([
#  y  z  x
  [1, 3, 0], # y
  [3, 1, 0], # z
  [0, 0, 0]  # x
])

eigenvalues = np.linalg.eigvals(A)
is_positive_semidefinite = np.all(eigenvalues >= 0)
print("Matrix A is positive semidefinite:", is_positive_semidefinite)
