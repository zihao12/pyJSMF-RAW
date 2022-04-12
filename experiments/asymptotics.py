import os
import sys
import pickle
import argparse
from scipy import sparse, io 
import numpy as np
import rpy2.robjects as robjects
import magic
import scprep
from sklearn.decomposition import NMF, non_negative_factorization, PCA
from sklearn.decomposition._nmf import _beta_divergence
import seaborn as sns
import pandas as pd
from scipy.sparse import csr_matrix, csc_matrix 


script_dir = "../"
sys.path.append(os.path.abspath(script_dir))
from file2 import *
from factorize import *
#from smallsim_functions import simulate_multinomial_counts
from misc import *
from findK_correction_jk import *

import pdb

np.random.seed(123)


def simulate_dir_multinom(F, alpha, d, n):
    L = np.random.dirichlet(alpha, n)
    Prob = L.dot(F.T)
    X = np.empty(shape = (n, p))
    for i in range(n):
        X[i,] = np.random.multinomial(d, Prob[i,], size = 1).astype(int)

    return X, L

n = 8000
p = 5000
d = 50
K = 50
n_sample = 100


## simulate F
F = np.random.normal(size = (p, K))
F = np.exp(F)
F /= F.sum(axis = 0)[None, :]
## set alpha
alpha = np.ones(K) / K
## get C
_, L = simulate_dir_multinom(F, alpha, d, n)
C = L @ F.T
C = C.T @ C / n
Cbar = C / C.sum(axis = 1)[:, None]
g = (Cbar**2).sum(axis = 1)



gs = np.zeros((p, n_sample))
for i in range(n_sample):
    print(i)
    X, _ = simulate_dir_multinom(F, alpha, d, n)
    if X.sum(axis = 0).min() < 1:
        print("words with 0 counts")
        pass
    
    C_ = compute_C_unbiased(X)
    Cbar_ = C_ / C_.sum(axis = 1)[:, None]
    g_ = (Cbar_**2).sum(axis = 1)
    gs[:, i] = g_

## get standardized sample

samples = (gs - g[:, None]) * np.sqrt(n*p*d)
rs = C.sum(axis = 0) * p
out = {'x': samples, 'r': rs, 'F': F}

outputfile=f"asymp_n{n}_p{p}_d{d}_K{K}.pkl"
with open(outputfile, "wb") as f:
    pickle.dump(out, f)