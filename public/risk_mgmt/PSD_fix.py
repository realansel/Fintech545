import scipy
import numpy as np

# implement Cholesky function in python
def cholesky(A):
    # A - must be positive definite covariance matrix
    n = len(A)
    L = np.zeros((n,n))
    for i in range(n):
        for j in range(i+1):
            s = sum(L[i][k] * L[j][k] for k in range(j))
            if i == j:
                L[i][j] = np.sqrt(A[i][i] - s)
            else:
                L[i][j] = (A[i][j] - s) / L[j][j]
    return L


# Implement chol_psd
def chol_psd(A):
  n = A.shape[1]
  root = np.zeros((n,n))
  
  # loop over columns
  for i in range(n):
    s = 0.0
    if i > 0:
      s = root[i][:i].T @ root[i][:i]

    
    # Diagonal Element
    temp = A[i][i] - s
    if temp <= 0 and temp >= -1e-8:
      temp = 0.0
    root[i][i] = np.sqrt(temp)

    # check for the 0 eign value. set the column to 0 if we have one
    if root[i][i] == 0.0:
      root[i][(i+1):n] = 0.0
    else:
      # update off diagonal rows of the column
      ir = 1.0/root[i][i]
      for j in np.arange(i+1,n):
        s = root[j][:i].T @ root[i][:i]
        root[j][i] = (A[j][i] -s) * ir
  return root

'''
Non PSD fixes
'''

# Implement near_psd
def near_psd(a, epsilon=0.0):
  n = a.shape[0]
  invSD = None
  out = a.copy()

  #calculate the correlation matrix if we got a covariance
  if (np.count_nonzero(np.diag(out) == 1.0) != n):
      invSD = np.diag(1.0 / np.sqrt(np.diag(out)))
      out = np.dot(np.dot(invSD, out), invSD)

  #SVD, update the eigen value and scale
  vals, vecs = np.linalg.eigh(out)
  vals = np.maximum(vals, epsilon)
  T = 1.0 / (np.dot(np.dot(vecs, np.diag(vals)), vecs.T))
  T = np.diag(np.sqrt(np.diag(T)))
  l = np.diag(np.sqrt(vals))
  B = np.dot(np.dot(T, vecs), l)
  out = np.dot(B, B.T)

  #Add back the variance
  if invSD is not None:
      invSD = np.diag(1.0 / np.diag(invSD))
      out = np.dot(np.dot(invSD, out), invSD)
  return out


# Create Higham's method that turn non pd to psd
def higham(R, tol=100):
  state = True
  while state is True:
    λ, V = scipy.linalg.eigh(R)
    check = 0
    for i in λ:
      if i < -1e-9:
        check +=1
    if check == 0 or tol <0:
      state = False
      break
    D = np.diag(np.maximum(λ,0))
    R = V @ D @ V.T
    tol -= 1
  return R

# Frobenius Norm
def f_norm(matrix):
  return np.linalg.norm(matrix, 'fro')

# Check the matrix is PSD or not
def is_psd(matrix):
    """For a given matrix, check if the matrix is psd or not."""
    eigenvalues = np.linalg.eigh(matrix)[0]
    return np.all(eigenvalues >= -1e-8)