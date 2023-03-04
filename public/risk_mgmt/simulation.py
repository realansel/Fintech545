import numpy as np
from risk_mgmt.PSD_fix import chol_psd


def direct_simulation(cov,nsim):
  result = chol_psd(cov) @ np.random.standard_normal(size=(len(cov), nsim))
  return result


def simulate_pca(a, nsim, perc=0.95):
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