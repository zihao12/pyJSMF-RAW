## simulate bows data from fitted
import os
import sys
from scipy import sparse
from scipy import sparse, io
import pdb


import numpy as np
import pickle
from sklearn.decomposition import NMF, non_negative_factorization
from sklearn.decomposition._nmf import _beta_divergence


script_dir = "../"
sys.path.append(os.path.abspath(script_dir))
from factorize import *
from misc import *
from findK_correction_jk import *

np.random.seed(123)

output_dir="../dataset/sim_bows"
fitted_dir="../output"

# data_name="nips"
# method="pn"
# k = 5

data_name = sys.argv[1]
k = int(sys.argv[2])
method="random"


## load fitted model
modelfile=f"{fitted_dir}/{data_name}_fitted_init_{method}_k{k}.Rds"
readRDS = robjects.r['readRDS']
fitted = readRDS(modelfile)
F = np.asarray(fitted.rx2('F'))
L = np.asarray(fitted.rx2('L'))

s = (L @ F.T).sum(axis = 1)
F, L = poisson2multinom(F.copy(), L.copy())

X = simulate_multinomial_counts(L, F, s)
X_val = simulate_multinomial_counts(L, F, s)
C = compute_C_unbiased(X)
X = sparse.csr_matrix(X.astype(int))
X_val = sparse.csr_matrix(X_val.astype(int))

io.mmwrite(f"{output_dir}/{data_name}_sim_k{k}_X.mtx", X)
io.mmwrite(f"{output_dir}/{data_name}_sim_k{k}_Xval.mtx", X_val)
np.savetxt(f"{output_dir}/{data_name}_sim_k{k}_C.csv", C, delimiter=',')



