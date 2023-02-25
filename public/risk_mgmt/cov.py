import numpy as np



def exp_covmatrix(matrix,lamb):
  # initialize cov matrix.
  n = matrix.shape[1]
  print(n)
  covMatrix = np.zeros((n,n))
  for i in range(len(matrix.columns)):
    for j in range(len(matrix.columns)):
      covMatrix[i][j] = exp_cov(matrix.iloc[:, i], matrix.iloc[:,j], n, lambda_value=lamb)
  return covMatrix


# Get covariance from correlation and variance
# We can easily get 4 suites of covariance estimation
def return_cov(corr, var):
  std_vector = np.sqrt(var)
  cov = corr * np.outer(std_vector, std_vector)
  return np.array(cov)



'''
Below are functions for functions
'''

# Expoentially weights
# n is length of data that need to weighted
def weight_f(n, lambda_value=0.94):
  w = np.empty(n, dtype=float)
  w = np.array([(1-lambda_value) * lambda_value**(i-1) for i in range(n)])
  w = sorted(w/np.sum(w))
  return w


# weighted variance

def exponential_weighted_variance(data,lambda_value):
    n = data.shape[0]
    m = data.shape[1]
    var = np.zeros((m,))
    weights = np.array([(1 - lambda_value) * lambda_value ** (i - 1) for i in range(n)])
    weights = sorted(weights/np.sum(weights))
    for i in range(m):
      xm = np.mean(data.iloc[:,i])
      var_sum = 0
      for j in range(n):
        var_sum += weights[n-1-j]*(data.iloc[:,i][n-1-j] - xm)**2
      var[i] = var_sum
    return var


# Function that calculate exponential weighted covariance
def exp_cov(X,Y, n,lambda_value = 0.94):
    cov = 0
    w = np.empty(n, dtype=float)
    w = np.array([(1-lambda_value) * lambda_value**(i-1) for i in range(n)])
    w = sorted(w/np.sum(w))
    xm = np.mean(X)
    ym = np.mean(Y)
    for i in range(n):
        cov += (w[n-i-1] * (X[n-i-1]-xm) * (Y[n-i-1]-ym))
    return cov