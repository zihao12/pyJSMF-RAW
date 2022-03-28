import numpy as np
import pdb

def findS_correction(X, Cbar, C, K):
    n, p = X.shape
    d = X.sum() / n
    w = X.sum(axis = 0)
    w /= w.sum()


    R = Cbar.T
    S = np.zeros(K)
    u = None
    diagR = np.zeros(K)

    g = (R**2).sum(axis = 0) ## naive estimate
    g = correction_step1(g, X, C, d, w) ## debias, compute variance, apply shrinkage

    for k in range(K):
        maxSquaredSum = np.max(g)
        maxCol = np.argmax(g)
        S[k] = maxCol
        u = R[:, maxCol].copy()
        u = u / np.sqrt(maxSquaredSum)
        diagR[k] = np.sqrt(maxSquaredSum)
        R = R -  u[:, None] @ (u[:,None].T @ R)

        dg = (Cbar @ u)**2
        dg = correction_stepk(dg, X, u, C, d, w) ## debias, compute variance, apply shrinkage
        g -= dg        
        
    return S, diagR

def correction_step1(g, X, C, d, w):
  n = X.shape[0]
  g -= 1 / (n*w * d**2) ## debias
  se = compute_se_step1(X, C, d, w)
  g = shrink(g, se)

  return g

def compute_se_step1(X, C, d, w):
  n = X.shape[0]
  D = d * (d-1)
  X4 = np.power(X, 4)
  X3 = np.power(X, 3)
  X2 = np.power(X, 2)
  ## compute variance part
  mu4 = (X4 * X4.sum(axis = 1)[:, None]).mean(axis = 0) / np.power(D, 4)
  mu4_mu =   (X4 * (X4 @ C.T)).mean(axis = 0) / np.power(D, 4)
  mu2_2 = np.power(X2.T @ X2, 2).sum(axis = 1) / np.power(n, 2) / np.power(D, 4)
  mu2_mu_2 = (X2 * (X2 @ np.power(C, 2))).mean(axis = 0) / np.power(D, 2)
  mu_4 = np.power(C, 4).sum(axis = 1)
  var = n * mu4 + 4*n*(n-1)*mu4_mu + (2*n*n - 3*n)*mu2_2 + 4 * n * (n-1) * (n-3) *mu2_mu_2 - 2* n * (n-1) * (2*n - 3)*mu_4
  #var = n * mu4 + 4*n*(n-1)*mu4_mu + (2*n*n - 3*n)*mu2_2 

  ## compute covariance
  # cov = 2 * (X3 * (X2.sum(axis = 1) * X.sum(axis = 1) - X3.sum(axis = 1))[:, None]).mean(axis = 0) / np.power(D, 3)
  cov = 2 * (X3 * (X2.sum(axis = 1)[:, None] * (X @ C.T) - X3 @ C.T)).mean(axis = 0) / np.power(D, 3)

  out = (var + cov) / np.power(n, 4)

  return np.sqrt(out / np.power(w, 4))


def correction_stepk(dg, X, u, C, d, w):
  dg -= compute_bias_stepk(X, u, d, w)
  se = compute_se_stepk(X, u, C, d, w)
  dg = shrink(dg, se)

  return dg

def compute_bias_stepk(X, u, d, w):
  n = X.shape[0]
  # out = np.power((X * (X @ u)[:, None] - X * u[None, :]), 2).mean(axis = 0) / n / np.power(d, 4)
  d2 = d * (d-1)
  out = (X @ u)[:, None] * X - X * u[None, :]
  out = (out / d2)**2
  out = out.mean(axis = 0) / n
  out /= np.power(w, 2)

  return out

def compute_se_stepk(X, u, C, d, w):
  n = X.shape[0]
  D = d * (d-1)
  mu = C @ u
  mu2 = np.power((X * (X @ u)[:, None] - X * u[None, :]), 2).mean(axis = 0) / np.power(D, 2)
  mu3 = np.power((X * (X @ u)[:, None] - X * u[None, :]), 3).mean(axis = 0) / np.power(D, 3)
  mu4 = np.power((X * (X @ u)[:, None] - X * u[None, :]), 4).mean(axis = 0) / np.power(D, 4)

  # out = n * mu4 + 4*n*(n-1)*mu3*mu + (2*n*n - 3*n)*np.power(mu2, 2) + 4*n*(n-1)*(n-3)*mu2*np.power(mu, 2) - 2*n*(n-1)*(2*n - 3)*np.power(mu, 4)
  out = n * mu4 + 4*n*(n-1)*mu3*mu + (2*n*n - 3*n)*np.power(mu2, 2)       
  #pdb.set_trace() ## the last 3 terms are larger!! second last is the largest
  out /= np.power(n, 4)
  
  return np.sqrt(out / np.power(w, 4))

def shrink(x, se):

  return x



def compute_C_unbiased_local(X):
    ## X is n by p
    n,  p = X.shape
    d = X.sum(axis = 1)
    m = (d * (d -1)).sum()
    C = (X.T @ X) - np.diag(X.sum(axis = 0))
    C = C / m
    
    return C



