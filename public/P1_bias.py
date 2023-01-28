import numpy as np
import scipy
from scipy import stats
from scipy.stats import skew
mu, sigma = 0,1

# Generating standard normal data set
sk_list = []

# Find 
for i in range(10000):
  s = np.random.normal(mu, sigma, 100)
  after = skew(s)
  sk_list.append(after)
m_skew = np.mean(sk_list)
std_skew = np.std(sk_list)
print(m_skew,std_skew)

# t statistics
t = (m_skew - 0)/ (std_skew/np.sqrt(10000))
scipy.stats.t.ppf(1-t, 99)
sk_list = np.array(sk_list)
print(stats.ttest_1samp(sk_list,0))


kur_list =[]
for i in range(10000):
  s = np.random.normal(mu, sigma, 100)
  after = stats.kurtosis(s)
  kur_list.append(after)
m_kurt = np.mean(kur_list)
std_kurt = np.std(kur_list)
# print(m_kurt,std_kurt)

# t stats
t = (m_kurt - 0)/ (std_kurt/np.sqrt(10000))
print(t)
print(stats.ttest_1samp(kur_list,0))