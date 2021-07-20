import numpy as np
import pdb

def simulate_factors(p, k, n_top = 20):
    F = np.random.uniform(size = (p, k))
    id_m = np.empty((n_top, k))
    anchor_words = np.empty(k)
    #pdb.set_trace()
    for i in range(k):
        idx = np.random.choice(a = range(p), size = n_top, replace=False)
        F[idx, i] = 100 * F[idx, i]
        anchor_words[i] = idx[np.argmax(F[idx, i])]
        id_m[:, i] = idx

    for i, a in enumerate(anchor_words.astype(int)):
        tmp = F[a,i]
        F[a,:] = np.repeat(0, k)
        F[a, i] = tmp   

    return F/F.sum(axis = 0), id_m.astype(int), anchor_words.astype(int)


def simulate_loadings(n, k, S):
    L = np.empty(shape = (n, k))
    for i in range(n):
        u = np.random.multivariate_normal(mean = np.zeros(k), cov = S)
        u = u - u.max()
        L[i,] = np.exp(u)/sum(np.exp(u))

    return L

def simulate_multinomial_counts(L, F, s):
    #pdb.set_trace()
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
    F, id_m, anchor_words = simulate_factors(p, k)
    L = simulate_loadings(n, k, S)
    X = simulate_multinomial_counts(L, F, s)

    return {'X':X, 'L':L, 'F':F, "id_m":id_m, "anchor_words":anchor_words}

def smallsim_correlated(n = 100, p = 400, k = 6, doc_len = 50):
    n = 100
    p = 400
    k = 6
    s = np.repeat(doc_len, n)
    S = 13 * np.diag(np.repeat(1, k)) - 2
    S[k-2, k-1] = 8
    S[k-1, k-2] = 8
    F, id_m, anchor_words = simulate_factors(p, k)
    L = simulate_loadings(n, k, S)
    X = simulate_multinomial_counts(L, F, s)

    return {'X':X, 'L':L, 'F':F, "id_m":id_m, "anchor_words":anchor_words}
