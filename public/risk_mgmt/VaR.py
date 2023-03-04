import numpy as np
import scipy.stats as stats
from risk_mgmt.cov import ew_covar

# Calculate Var
def calculate_var(data, mean=0, alpha=0.05):
  return mean - np.quantile(data, alpha)


# Calculate ES
def calculate_es(data, mean=0, alpha=0.05):
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
    ew_cov = ew_covar(data, lambd)
    
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
