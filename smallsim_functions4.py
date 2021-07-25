import numpy as np

def simulate_factors(p, k, n_top = 20):
    F = np.random.uniform(size = (p, k))
    for i in range(k):
        F[(i*n_top):(i*n_top + n_top), i] *= 100

    return F/F.sum(axis = 0), k*n_top


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

    return X

def smallsim_independent(n = 100, p = 400, k = 6, doc_len = 50):
    s = np.repeat(doc_len, n)
    S = 13 * np.diag(np.repeat(1, k)) - 2
    F, p0 = simulate_factors(p, k)
    L = simulate_loadings(n, k, S)
    X = simulate_multinomial_counts(L, F, s)

    A = (L.T @ L) / n

    return X, A, F, p0

