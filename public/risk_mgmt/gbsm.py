import numpy as np
from math import log, sqrt, exp
from scipy.stats import norm

def gbsm_greeks(S, K, tau, r, q, sigma, option_type='call'):
    d1 = (log(S / K) + (r - q + 0.5 * sigma**2) * tau) / (sigma * sqrt(tau))
    d2 = d1 - sigma * sqrt(tau)
    N_d1 = norm.cdf(d1)
    N_d2 = norm.cdf(d2)
    if option_type.lower() == 'call':
        delta = exp(-q * tau) * N_d1
        gamma = exp(-q * tau) * norm.pdf(d1) / (S * sigma * sqrt(tau))
        vega = S * np.exp(-q * tau) * norm.pdf(d1) * np.sqrt(tau)
        theta = -S * np.exp(q * tau) * norm.pdf(d1) * sigma / (2 * np.sqrt(tau)) - r * K * np.exp(-r * tau) * norm.cdf(-d2) + q * S * np.exp(-q * tau) * norm.cdf(-d1)
        rho = K * tau * np.exp(-r * tau) * norm.cdf(d2)
        P = S * np.exp(-q * tau) * norm.cdf(d1) - K * np.exp(-r * tau) * norm.cdf(d2)
    elif option_type.lower() == 'put':
        delta = -exp(-q * tau) * (1 - N_d1)
        gamma = exp(-q * tau) * norm.pdf(d1) / (S * sigma * sqrt(tau))
        vega = S * np.exp(-q * tau) * norm.pdf(d1) * np.sqrt(tau)
        theta = -S * np.exp(q * tau) * norm.pdf(d1) * sigma / (2 * np.sqrt(tau)) + r * K * np.exp(-r * tau) * norm.cdf(-d2) - q * S * np.exp(-q * tau) * norm.cdf(-d1)
        rho = -K * tau * np.exp(-r * tau) * norm.cdf(-d2)
        P = K * np.exp(-r * tau) * norm.cdf(-d2) - S * np.exp(-q * tau) * norm.cdf(-d1)
    else:
        raise ValueError("option_type must be 'call' or 'put'")
    return {'P':P, 'delta':delta, 'gamma':gamma, 'vega':vega, 'theta':theta, 'rho':rho}