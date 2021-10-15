## implement sinkhorn normalization
import numpy as np

def sinkhorn(X, niter = 10):
  r = X.mean(axis = 1)
  for i in range(niter):   
      c = 1/(X * r[:, None]).mean(axis = 0) 
      r = 1/(X * c).mean(axis = 1)       
  Xnorm = np.sqrt(r)[:, None] * X * np.sqrt(c)
  
  return Xnorm, r, c


# Inputs:
# - X: n by p data matrix (possibly normalized)
# - r: number of eigenvectors to use 
# Outputs:
# - C: p by p covariance matrix
def X2C_svd(X, r=50):
	n, p = X.shape
	u, s, vh  = np.linalg.svd(X/np.sqrt(n), full_matrices=False)
	l = (X.sum(axis = 1) ** 2).mean()
	C =  vh[:r, :].T @ np.diag(s[:r]**2) @ vh[:r, :] - np.diag(np.repeat(1, p))
	C /= l
	C[C < 0] = 1e-16
	C /= C.sum()

	return C
