import numpy as np
import pdb

def compute_C_unbiased_local(X):
    ## X is n by p
    n,  p = X.shape
    d = X.sum(axis = 1)
    m = (d * (d -1)).sum()
    C = (X.T @ X) - np.diag(X.sum(axis = 0))
    C = C / m
    
    return C

def findS_correction(Cbar, C, K, const):
    R = Cbar.T.copy()## why? I change Cbar without .copy()
    S = np.zeros(K)
    u = None
    colSquaredSums = (R**2).sum(axis = 0)
    diagR = np.zeros(K)

    for k in range(K):
        colSquaredSums -= correction(C, u, const)
        maxSquaredSum = np.max(colSquaredSums)
        maxCol = np.argmax(colSquaredSums)
        S[k] = maxCol
        u = R[:, maxCol]
        u = u / np.sqrt(maxSquaredSum)
        diagR[k] = np.sqrt(maxSquaredSum)
        colSquaredSums -= (u.T @ R)**2
        
        R = R -  u[:, None] @ (u[:,None].T @ R)
        
    return S, diagR

def Cbar_pj_rss(C, u, weights, const):
    C_ = C.copy()
    Cbar = C_ / C_.sum(axis = 1)[:, None]
    out = (Cbar**2).sum(axis = 1) - (Cbar @ u)**2
    out -= correction(C_, None, const)
    out -= correction(C_, u, const)
    
    return out

def est_utC_sq(C, d, u, X):
    D = d*d - d
    u2 = u**2
    X2 = X**2
    m = sum(D)
    weights = C.sum(axis = 0)
    out = (C @ u) ** 2 

    ## original code ###################
    out -= (u**2) * weights * d.sum() / (m**2)
    out -=  ( C @ u2 + 2 * (C @ u) * u - 3 * u2 * np.diag(C) ) / m
    #pdb.set_trace()
    tmp = ((X @ u)**2 - X2 @ u2)[:, None] - 2* (X @ u)[:, None] * (X * u[None, :]) + 2* X2 * u2[None, :]
    tmp = X2 * tmp
    ## original code ###################

    # ## new code #####
    # tmp = X2 * ((X @ u)**2)[:, None]
    # ## new code #####


    tmp = tmp / (d**3)[:, None]
    tmp = tmp.mean(axis = 0)
    tmp = tmp * (d**3).sum()
    tmp = tmp / (m**2)
    out = out - tmp

    out /= (1 - np.sum(D**2) / m**2) ## changes very little

    return out


def correction(C, u, const):
    weights = C.sum(axis = 1)
    if u is None:
        return const / weights
    return - const * (C @ (u**2)) / (weights**2) - const * (C @ u)**2 / (weights**2)



def Cbar_rss(Cbar, weights, const):
    weights_inv = 1/weights
    rss_est = compute_rss_mean(Cbar, const, weights_inv)
    rsq_est = compute_rsq_mean(Cbar, rss_est, const, weights_inv)
    rss_var = compute_rss_var(rss_est, rsq_est, const, weights_inv)
    
    return rss_est, rss_var
    
def compute_rss_var(rss_est, rsq_est, const, weights_inv):
    rss_var = (const**3) * (weights_inv**3) + 6 * (const**2) * (weights_inv**2) * rss_est 
    rss_var += 4 * const * weights_inv * rsq_est
    
    return rss_var

def compute_rss_mean(Cbar, const, weights_inv):
    rss_est = (Cbar**2).sum(axis = 1) - const * weights_inv
    
    return rss_est

def compute_rsq_mean(Cbar, rss_est, const, weights_inv):
    rsq_est = (Cbar**3).sum(axis = 1) - 3 * const * weights_inv * rss_est - (const*weights_inv)**2
    
    return rsq_est

def simulate_multinomial_counts_local(L, F, s):
    n = L.shape[0]
    p = F.shape[0]
    Prob = L.dot(F.T)
    X = np.empty(shape = (n, p))
    for i in range(n):
        X[i,] = np.random.multinomial(s[i], Prob[i,], size = 1).astype(int)

    return X
