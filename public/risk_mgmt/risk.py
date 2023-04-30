from scipy.optimize import minimize
import numpy as np

def max_sharpe_ratio_weights(cov_m, exp_returns, rf, restrict="True"):
    num_stocks = len(exp_returns)
    
    # Define the Sharpe Ratio objective function to be minimized
    def neg_sharpe_ratio(weights):
        port_return = np.dot(weights, exp_returns)
        port_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_m, weights)))
        sharpe_ratio = (port_return - rf) / port_volatility
        return -sharpe_ratio
    
    # Define the constraints
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1}) # The sum of the weights must be 1
    if restrict == "True":
        bounds = tuple([(0, 1) for i in range(num_stocks)]) # The weights must be between 0 and 1
        initial_weights = np.ones(num_stocks) / num_stocks # Start with equal weights
        opt_results = minimize(neg_sharpe_ratio, initial_weights, method='SLSQP', bounds=bounds, constraints=constraints)
    elif restrict == "False":
        initial_weights = np.ones(num_stocks) / num_stocks # Start with equal weights
        opt_results = minimize(neg_sharpe_ratio, initial_weights, method='SLSQP', constraints=constraints)
    

    return opt_results.x.round(4), -opt_results.fun


def risk_parity_weights(covar):
    n = covar.shape[0]

    def pvol(x):
        return np.sqrt(x.T @ covar @ x)

    def pCSD(x):
        p_vol = pvol(x)
        csd = x * (covar @ x) / p_vol
        return csd

    def sseCSD(x):
        csd = pCSD(x)
        mCSD = np.sum(csd) / n
        dCsd = csd - mCSD
        se = dCsd * dCsd
        return 1.0e5 * np.sum(se)

    # Constraints
    cons = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1})

    # Bounds
    bnds = [(0, None) for _ in range(n)]

    # Initial guess
    x0 = np.array([1/n] * n)

    res = minimize(sseCSD, x0, method='SLSQP', bounds=bnds, constraints=cons)

    return np.round(res.x, decimals=4)



def covar_m(cor, vol):
    return np.diag(vol) @ cor @ np.diag(vol)

def risk_contribution(weights, cov):
    return weights * (cov @ weights / (np.sqrt(weights.T @ cov @ weights)))
