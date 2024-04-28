import numpy as np

# A = np.array([
#   [0.5, 0, 0],
#   [0, 0.5, 0],
#   [0, 0, -2]
# ])

A = np.array([
#  x  y, z
  [0, 0, 0], # x
  [0, 0, 0], # y
  [0, 0, 0]  # z
])

eps = 1e-9
eigenvalues = np.linalg.eigvals(A)
is_positive_semidefinite = np.all(eigenvalues >= -eps)
print("Matrix A is positive semidefinite:", is_positive_semidefinite)
