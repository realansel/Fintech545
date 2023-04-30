import numpy as np
import pandas as pd

def return_calculate(prices, method='DISCRETE', date_column='Date'):
    vars = prices.columns
    n_vars = len(vars)
    
    if date_column is not None:
        vars = [v for v in vars if v != date_column]
        if n_vars == len(vars):
            raise ValueError(f'{date_column} not in DataFrame {vars}')
    
    n_vars = len(vars)
    p = prices[vars].to_numpy()
    n, m = p.shape
    p2 = np.empty((n - 1, m))
    
    for i in range(n - 1):
        p2[i, :] = p[i + 1, :] / p[i, :]
    
    if method.upper() == 'DISCRETE':
        p2 -= 1.0
    elif method.upper() == 'LOG':
        p2 = np.log(p2)
    else:
        raise ValueError(f'method {method} must be in ("LOG", "DISCRETE")')
    
    if date_column is not None:
        dates = prices.iloc[1:, prices.columns.get_loc(date_column)]
        out = pd.DataFrame({date_column: dates})
    else:
        out = pd.DataFrame()
    
    for i in range(n_vars):
        out[vars[i]] = p2[:, i]
    
    return out

def get_portfolio_price(portfolio,prices,symbol):
    if symbol != "ALL":
        returns = prices.pct_change().dropna(how='all')
        pv = []
        for stock in portfolio[portfolio['Portfolio']==symbol]['Stock']:
            pv.append(prices.iloc[-1][stock])
        returns_p=[]
        for stock in portfolio[portfolio['Portfolio']==symbol].loc[:, 'Stock'].tolist():
            returns_p.append((returns.loc[:, stock]).tolist())
        returns_p=pd.DataFrame(returns_p).T
        holdings = portfolio[portfolio['Portfolio']==symbol]
        return pv, returns_p, holdings
    else:
        returns = prices.pct_change().dropna(how='all')
        pv = []
        for stock in portfolio['Stock']:
            pv.append(prices.iloc[-1][stock])
        returns_p=[]
        for stock in portfolio.loc[:, 'Stock'].tolist():
            returns_p.append((returns.loc[:, stock]).tolist())
        returns_p=pd.DataFrame(returns_p).T
        holdings = portfolio
        return pv, returns_p, holdings