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


def simulate_multinomial_counts_local(L, F, s):
    n = L.shape[0]
    p = F.shape[0]
    Prob = L.dot(F.T)
    X = np.empty(shape = (n, p))
    for i in range(n):
        X[i,] = np.random.multinomial(s[i], Prob[i,], size = 1).astype(int)

    return X

datafile="../../ebpmf_data_analysis/output/fastTopics_fit/sla_small_fitted.Rds"
Yfile = "../../ebpmf_data_analysis/output/fastTopics_fit/sla_small.txt"
vocabfile="../../ebpmf_data_analysis/output/fastTopics_fit/sla_small_vocab.txt"
titlefile="../../ebpmf_data_analysis/data/SLA/title.sla.txt"

# Y = io.mmread(Yfile)
vocab = np.loadtxt(vocabfile, dtype = str)
readRDS = robjects.r['readRDS']
data = readRDS(datafile)

w_idx = np.asarray(data.rx2('word_idx')).astype(int)
fitted = data.rx2('fit_sub')
F = np.asarray(fitted.rx2('F'))
L = np.asarray(fitted.rx2('L'))
s = np.asarray(fitted.rx2('s'))

p, k = F.shape
n = L.shape[0]
C = L @ F.T
C = C.T @ C / n
Cbar = C / C.sum(axis = 1)[:, None]
S, U = get_U(Cbar, k)
print(S)

truth = (Cbar**2).sum(axis = 1) 

w_true = C.sum(axis = 1)
## input: L, F, s, idx
n_sample = 50
est_notcorrected = np.zeros((p, k, n_sample))
est = np.zeros((p, k, n_sample))
se = np.zeros((p, k, n_sample))


for i in range(n_sample):
    print(i)
    s_ = np.random.poisson(lam = s.mean(), size = s.shape)
    X = simulate_multinomial_counts_local(L, F, np.round(s_))
    if X.sum(axis = 0).min() < 1:
        pass
    # w = X.sum(axis = 0)
    # w /= w.sum()
    # w = np.array(w)
    # pdb.set_trace()
    C_ = compute_C_unbiased(X)
    Cbar_ = C_ / C_.sum(axis = 1)[:, None]
    gs, vs, gs_notcorrected = exper_jk(X, Cbar_, C_, U)
    #pdb.set_trace()
    est[:, :, i] = gs 
    se[:, :, i] = np.sqrt(vs)
    est_notcorrected [:, :, i] = gs_notcorrected
 
outputfile="findK-exper-jk2.pkl"
with open(outputfile, "wb") as f:
    pickle.dump({'est':est, 'est_notcorrected':est_notcorrected,
        'se':se, 'C':C, 'truth': truth}, f)