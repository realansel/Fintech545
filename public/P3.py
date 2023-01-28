from statsmodels.tsa.arima_process import arma_generate_sample
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

np.random.seed(12)

# function that plot acf and pacf
def plot_acfa(y):
    fig, ax = plt.subplots(1,2,figsize=(10,5))
    plot_acf(y, ax=ax[0],lags=10)
    plot_pacf(y, ax=ax[1], lags=10, method='ywm')
    plt.show()


# AR(1)
ar_coefs = [1, -0.5]

y = arma_generate_sample(ar_coefs, [1], scale=0.1, nsample=1000, burnin=50)
plt.plot(y)
plt.title("AR(1)")
plt.show()
plot_acfa(y)
print(np.mean(y), np.std(y))

# AR(2)
ar_coefs = [1, 0.5, -0.2]
y = arma_generate_sample(ar_coefs, [1], 1000)
plt.plot(y)
plt.title("AR(2)")
plt.show()
plot_acfa(y)

# AR(3)
ar_coefs = [1, 0.5, 0.3, -0.2]
y = arma_generate_sample(ar_coefs, [1], 1000)
plt.plot(y)
plt.title("AR(3)")
plt.show()
plot_acfa(y)

# MA(1)
ma_coefs = [1, 0.5]
y = arma_generate_sample([1], ma_coefs, 1000)
plt.plot(y)
plt.title("MA(1)")
plt.show()
plot_acfa(y)

# MA(2)
ma_coefs = [1, -0.3, 0.2]
y = arma_generate_sample([1], ma_coefs, 1000)
plt.plot(y)
plt.title("MA(2)")
plt.show()
plot_acfa(y)

# MA(3)
ma_coefs = [1, 0.5, -0.2, 0.1]
y = arma_generate_sample([1], ma_coefs, 1000)
plt.plot(y)
plt.title("MA(3))")
plt.show()
plot_acfa(y)