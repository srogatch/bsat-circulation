import numpy as np

# A = np.array([
#   [0.5, 0, 0],
#   [0, 0.5, 0],
#   [0, 0, -2]
# ])

A = np.array([
#  x  y, z
  [-1, -1, -1], # x
  [-1, -1, -1], # y
  [-1, -1, -1]  # z
])

eps = 1e-9
eigenvalues = np.linalg.eigvals(A)
is_positive_semidefinite = np.all(eigenvalues >= -eps)
print("Matrix A is positive semidefinite:", is_positive_semidefinite)
