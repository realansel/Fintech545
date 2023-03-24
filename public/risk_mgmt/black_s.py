import numpy as np
import scipy.stats as stats
import pandas as pd


def black_scholes(S, X, r, q, t, sigma, option):
    # S = current stock price
    # X = option strike price
    # t = time to maturity
    # r = risk-free rate
    # q = continuously compounding dividend yield
    # sigma = volatility
    
    d1 = (np.log(S / X) + (r - q + (sigma ** 2) / 2) * t) / (sigma * np.sqrt(t))
    d2 = d1 - sigma * np.sqrt(t)
    if option.lower() == "call":
        P = S * np.exp(-q * t) * stats.norm.cdf(d1) - X * np.exp(-r * t) * stats.norm.cdf(d2)
    elif option.lower() == 'put':
        P = X * np.exp(-r * t) * stats.norm.cdf(-d2) - S * np.exp(-q * t) * stats.norm.cdf(-d1)
    else:
        raise ValueError("Choose option type from 'put' and 'call'.")
    
    return P




def implied_volatility(S, X, r, q, t, market_price, option_type='call', initial_guess=0.5, max_iter=300, tol=1e-8):
    sigma = initial_guess

    for _ in range(max_iter):
        if option_type.lower() == 'call':
            option_price = black_scholes(S, X, r, q, t, sigma, 'call')
        elif option_type.lower() == 'put':
            option_price = black_scholes(S, X, r, q, t, sigma, 'put')
        else:
            raise KeyError('InValid option type, choose "put" or "call".')

        d1 = (np.log(S / X) + (r - q + (sigma ** 2) / 2) * t) / (sigma * np.sqrt(t))
        vega = S * np.exp(-q * t) * stats.norm.pdf(d1) * np.sqrt(t)

        diff = option_price - market_price

        if abs(diff) < tol:
            return sigma

        sigma = sigma - diff / vega

    return None  # implied volatility not found within the given tolerance


# function add Time to maturity and implied volatility to a portfolio csv file

def add_VolnT(S, q, rf, portfolio, current_date):
    current_date = pd.to_datetime(current_date)
    portfolio['ExpirationDate'] = pd.to_datetime(portfolio['ExpirationDate'])
    portfolio['T'] = (portfolio['ExpirationDate'] - current_date).dt.days / 365

    portfolio['Implied_Volatility'] = portfolio.apply(
        lambda row: implied_volatility(S, row['Strike'], rf, q, row['T'], row['CurrentPrice'], row['OptionType'])
        if row['Type'] == 'Option' else np.nan,
        axis=1
    )
    return portfolio


def portfolio_value(portfolio, underlying_range, r, q, current_date, day_ahead = 0):
    values = []

    for underlying in underlying_range:
        total_value = 0

        for _, row in portfolio.iterrows():
            holding = row['Holding']

            if row['Type'] == 'Stock':
                total_value += holding * underlying
            elif row['Type'] == 'Option':
                T = ((row['ExpirationDate'] - current_date) / pd.Timedelta(1, 'D') - day_ahead) / 365
                option_value = black_scholes(underlying, row['Strike'], r, q, T, row['Implied_Volatility'], row['OptionType'])
                total_value += holding * option_value
        
        values.append(total_value)

    return values