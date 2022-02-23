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
from smallsim_functions import simulate_multinomial_counts
from misc import *
from findK_correction import *

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
s = np.repeat(np.round(s.mean()), s.shape[0]) ## equal length

p, k = F.shape
n = L.shape[0]
C = L @ F.T
C = C.T @ C / n
Cbar = C / C.sum(axis = 1)[:, None]
truth = (Cbar**2).sum(axis = 1) 

## input: L, F, s, idx
n_sample = 200
est = np.zeros((p, n_sample))
est_noncorrected = np.zeros((p, n_sample))
var = np.zeros((p, n_sample))


const =  1 / (s*s - s).sum()


for i in range(n_sample):
    print(i)
    X = simulate_multinomial_counts_local(L, F, np.round(s))
    if X.sum(axis = 0).min() < 1:
        pass
    C_ = compute_C_unbiased_local(X)
    weights_inv = 1/C_.sum(axis = 1)
    Cbar_ = C_ / C_.sum(axis = 1)[:, None]
    
    rss_est = compute_rss_mean(Cbar_, const, weights_inv)
    rsq_est = compute_rsq_mean(Cbar_, rss_est,const, weights_inv)
    rss_var = compute_rss_var(rss_est, rsq_est, const, weights_inv)
    
    est[:, i] = rss_est
    est_noncorrected[:, i] = (Cbar_**2).sum(axis = 1)
    var[:, i] = rss_var
    
outputfile="findK-exper-step1-constd.pkl"
with open(outputfile, "wb") as f:
    pickle.dump({'est':est, 'est_noncorrected':est_noncorrected,
        'var':var, 'C':C, 'truth': truth}, f)
