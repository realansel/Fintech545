import pandas as pd
import numpy as np
from risk_mgmt import return_calculate, exp_covmatrix
data = pd.read_csv("./DailyPrices.csv")
data_return = return_calculate(data)
data_return.drop('Date', axis=1, inplace=True)
print(exp_covmatrix(data_return))