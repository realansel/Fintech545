import statsmodels.api as sm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy
from scipy.optimize import minimize
from scipy.special import gammaln
import math

file = pd.read_csv("/Users/ansel_li/Fintech545/public/Week02/problem2.csv")
X=file.x
Y=file.y
model = sm.OLS(Y,X)
result = model.fit()
beta = result.params[0]
print(result.summary())
error = Y - X * beta
#print(error)

fig, (ax2, ax3) = plt.subplots(1, 2, figsize=(10, 4))

ax2.plot(X,Y,'ro')
ax2.title.set_text("Data distribution and regression line")
x = np.linspace(-5,5,100)
y = x * beta
ax2.plot(x,y)
scipy.stats.probplot(error, dist="norm", plot=ax3)
ax3.title.set_text("Q-Q plot")
plt.show()

mu, std = scipy.stats.norm.fit(error)
print("Mean: ", mu)
print("Standard Deviation: ", std)



# MLE function
# ml modeling and neg LL calculation
def MLE_Norm(parameters):
  # extract parameters
  const, beta, std_dev = parameters
  # predict the output
  pred = const + beta*X
  # Calculate the log-likelihood for normal distribution
  LL = np.sum(scipy.stats.norm.logpdf(Y, pred, std_dev))
  # Calculate the negative log-likelihood
  neg_LL = -1*LL
  return neg_LL

mle_model = minimize(MLE_Norm, [1,1,1])
beta_hat = mle_model.x[1]
like = mle_model.fun
print(beta_hat)
print("AIC: ", 6 + like * 2)


def __T_loglikelihood(params):
  # extract parameters
  beta, std_dev = params
  # predict the output
  err = Y - beta*X
  # Calculate the log-likelihood for normal distribution
  LL = np.sum(scipy.stats.t.logpdf(err, 6.2, 0, std_dev))
  # Calculate the negative log-likelihood
  neg_LL = -1*LL# Return negative log-likelihood for optimization
  return neg_LL
# Initialize starting parameters
params_init = [1, 1]

# Perform MLE optimization
res = minimize(__T_loglikelihood, params_init)

# Extract fitted parameters
mu_hat, s_hat = res.x
AICp = 6 + res.fun * 2
# Print results
print(f'Fitted parameters: mu = {mu_hat}, s = {s_hat}')
print("AIC: ", AICp)
error2 = Y - X * s_hat
plt.plot(error2)
plt.title("Error visulization")
plt.show()
