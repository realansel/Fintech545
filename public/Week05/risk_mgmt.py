import numpy as np
import pandas as pd
import scipy
import scipy.stats as stats

def return_calculate(price, method = 'discrete'):
    returns = []
    for i in range(len(price)-1):
        returns.append(price[i+1]/price[i])
    returns = np.array(returns)
    if method == 'discrete':
        return returns - 1
    if method == 'log':
        return np.log(returns)



def exp_covmatrix(data,weight):
  # initialize cov matrix.
    data = data - data.mean(axis=0)
    weight = np.diag(weight)
    data_left = weight@data
    data_right = np.dot(data.T,data_left)
    return data_right

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
def exp_cov(X, Y, n, lambda_value):
    if len(X) < n or len(Y) < n:
        return 0
    w = np.ones(n) * lambda_value
    w[0] = 1 - lambda_value
    xm = np.mean(X)
    ym = np.mean(Y)
    cov = 0
    for i in range(n):
        if (n-i-1) < len(X) and (n-i-1) < len(Y):
            cov += (w[n-i-1] * (X[n-i-1]-xm) * (Y[n-i-1]-ym))
    return cov




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
      if i < -1e-8:
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





def direct_simulation(cov,nsim):
  result = chol_psd(cov) @ np.random.standard_normal(size=(len(cov), nsim))
  return result


def simulate_pca(a, nsim, perc):
    # a - matrix input
    # nsim - # of simulation
    # perc - percentage of explaination you want

    # Eigenvalue decomposition
    vals, vecs = np.linalg.eig(a)

    # flip the eigenvalues and the vectors
    flip = np.argsort(vals)[::-1]
    vals = vals[flip]
    vecs = vecs[:, flip]

    tv = np.sum(vals)
    start = 0
    while (np.abs(np.sum(vals[:start])/tv) <perc):
      start+=1
    vals = vals[:start]
    vecs = vecs[:, :start]
    print("Simulating with", start, "PC Factors: {:.2f}".format(np.abs(sum(vals)/tv*100)), "% total variance explained")
    B = np.matmul(vecs, np.diag(np.sqrt(vals)))
    m = B.shape[1]
    r = np.random.randn(m,nsim)
    return np.matmul(B, r)





# Calculate Var
def calculate_var(data, mean=0, alpha=0.05):
  return mean - np.quantile(data, alpha)


# Calculate ES
def calculate_es(data, mean, alpha):
  return -np.mean(data[data <= -calculate_var(data, mean, alpha)])


# Calculate Normal distribution VaR
def normal_var(data, mean=0, alpha=0.05, nsamples=10000):
    """
    Calculates the VaR for a normal distribution given a dataset, mean, confidence level, and number of samples.
    
    Args:
        data: a numpy array of data points
        mean: a float representing the mean of the data (default is 0)
        alpha: a float between 0 and 1 representing the desired confidence level (default is 0.05)
        nsamples: an integer representing the number of samples to use for the Monte Carlo simulation (default is 10000)
    
    Returns:
        The VaR for the normal distribution at the specified confidence level and mean.
    """
    sigma = np.std(data)
    simulation_norm = np.random.normal(mean, sigma, nsamples)
    var_norm = calculate_var(simulation_norm, mean, alpha)
    return var_norm


def ewcov_normal_var(data, mean=0, alpha=0.05, nsamples=10000, lambd = 0.94):
    """
    Calculates the Value at Risk (VaR) for a normal distribution using exponentially weighted covariance matrix.
    
    Args:
        data: a numpy array of data points
        mean: a float representing the mean of the data (default is 0)
        alpha: a float between 0 and 1 representing the desired confidence level (default is 0.05)
        nsamples: an integer representing the number of samples to use for the Monte Carlo simulation (default is 10000)
        lambd: a float between 0 and 1 representing the decay factor for the exponentially weighted covariance matrix (default is 0.94)
    
    Returns:
        The VaR for the normal distribution using exponentially weighted covariance matrix at the specified confidence level and mean.
    """
    
    # Calculate the exponentially weighted covariance matrix
    weights = weight_f(len(data), lambda_value=lambd)
    ew_cov = exp_covmatrix(data, weights)
    
    # Calculate the exponentially weighted variance and standard deviation
    ew_variance = ew_cov
    sigma = np.sqrt(ew_variance)
    simulation_ew = np.random.normal(mean, sigma, nsamples)
    var_ew = calculate_var(simulation_ew, mean, alpha)
    return var_ew



def MLE_t_var(data, mean=0, alpha=0.05, nsamples=10000):
    """
    Calculates the Value at Risk (VaR) for a t-distribution using a Monte Carlo simulation.
    
    Args:
        data: a numpy array of data points
        mean: a float representing the mean of the data (default is 0)
        alpha: a float between 0 and 1 representing the desired confidence level (default is 0.05)
        nsamples: an integer representing the number of samples to use for the Monte Carlo simulation (default is 10000)
    
    Returns:
        The VaR for the t-distribution at the specified confidence level and mean.
    """
    
    # Fit the t-distribution to the data
    params = stats.t.fit(data, method="MLE")
    df, loc, scale = params
    
    # Generate samples from the t-distribution
    simulation_t = stats.t(df, loc, scale).rvs(nsamples)
    
    # Calculate the VaR using the Monte Carlo simulation
    var_t = calculate_var(simulation_t, mean, alpha)
    
    return var_t


def historic_var(data, mean=0, alpha=0.05):
    return calculate_var(data, mean, alpha)