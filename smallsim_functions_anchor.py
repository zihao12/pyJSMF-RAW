import numpy as np
import pdb

def simulate_factors(p, k):
    F = np.random.uniform(size = (p, k))
    F[:k,:] = np.diag(np.repeat(5, k))


    return F/F.sum(axis = 0)


def simulate_loadings(n, k, S):
    L = np.empty(shape = (n, k))
    for i in range(n):
        u = np.random.multivariate_normal(mean = np.zeros(k), cov = S)
        u = u - u.max()
        L[i,] = np.exp(u)/sum(np.exp(u))

    return L

def simulate_multinomial_counts(L, F, s):
    n = L.shape[0]
    p = F.shape[0]
    Prob = L.dot(F.T)
    X = np.empty(shape = (n, p))
    for i in range(n):
        X[i,] = np.random.multinomial(s[i], Prob[i,], size = 1).astype(int)

    w_idx = np.where(X.sum(axis = 0) > 0)[0]
    X = X[:,w_idx]
    w_dict = {}
    #pdb.set_trace()
    for i, w in enumerate(w_idx):
        w_dict[w] = i

    F = F[w_idx,:] / F[w_idx,:].sum(axis = 0)

    return X, w_dict, F

def smallsim_independent(n = 100, p = 400, k = 6, doc_len = 50):
    s = np.repeat(doc_len, n)
    S = 13 * np.diag(np.repeat(1, k)) - 2
    F = simulate_factors(p, k)
    L = simulate_loadings(n, k, S)
    X, w_dict, F = simulate_multinomial_counts(L, F, s)

    return {'X':X, 'L':L, 'F':F}

def smallsim_correlated(n = 100, p = 400, k = 6, doc_len = 50):
    n = 100
    p = 400
    k = 6
    s = np.repeat(doc_len, n)
    S = 13 * np.diag(np.repeat(1, k)) - 2
    S[k-2, k-1] = 8
    S[k-1, k-2] = 8
    F, id_m = simulate_factors(p, k)
    L = simulate_loadings(n, k, S)
    X, w_dict, F = simulate_multinomial_counts(L, F, s)

    return {'X':X, 'L':L, 'F':F}

