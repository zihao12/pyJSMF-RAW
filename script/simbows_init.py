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

dir_name="../dataset/sim_bows"

# data_name="nips"
# method="pn"
# k = 5

data_name = sys.argv[1]
method = sys.argv[2]
k = int(sys.argv[3])


X = io.mmread(f"{dir_name}/{data_name}_sim_k{k}_X.mtx").astype(float).toarray()
C = np.loadtxt(f"{dir_name}/{data_name}_sim_k{k}_C.csv", delimiter=',')
Cbar = C / C.sum(axis = 1)[:, None]
C_rowSums = C.sum(axis=1)


n, p = X.shape
out = {}
outputfile=f"../output/{data_name}_sim_init_{method}_k{k}.pkl"

if method == "pn":
		S, diagR = findS_correction_jk(X.copy(), Cbar.copy(), C.copy(), k, prior_family = "point_normal")
		F, _, _ = recoverB(Cbar, C_rowSums, S, "activeSet")
if method == "rec1_pn": ## use rectification in step 1
		Cr, _, _ = rectifyC(C, k, "AP")
		Cr_bar = Cr / Cr.sum(axis = 1)[:, None]
		S, diagR = findS_correction_jk(X.copy(), Cr_bar, Cr, k, prior_family = "point_normal")
		F, _, _ = recoverB(Cbar, C_rowSums, S, "activeSet")

if method == "rec2_pn":## use rectification in step 2
		Cr, _, _ = rectifyC(C, k, "AP")
		Cr_bar = Cr / Cr.sum(axis = 1)[:, None]
		S, diagR = findS_correction_jk(X.copy(), Cbar, C, k, prior_family = "point_normal")
		print(S)
		F, _, _ = recoverB(Cr_bar, Cr.sum(axis = 1), S, "activeSet")

if method == "rec12_pn":## use rectification in step 1 & 2
		Cr, _, _ = rectifyC(C, k, "AP")
		Cr_bar = Cr / Cr.sum(axis = 1)[:, None]
		S, diagR = findS_correction_jk(X.copy(), Cr_bar, Cr, k, prior_family = "point_normal")
		F, _, _ = recoverB(Cr_bar, Cr.sum(axis = 1), S, "activeSet")

if method == "vanila":
		S, F, A, Btilde, _, _, _, _ = factorizeC(C, K=k, rectifier='no', optimizer='activeSet')

if method == "rec":
		S, F, A, Btilde, _, _, _, _ = factorizeC(C, K=k, rectifier='yes', optimizer='activeSet')

if method == "nndsvd":
		model_early = NMF(n_components=k, init='nndsvda', random_state=0,
		            beta_loss='kullback-leibler', solver='mu', max_iter=1, verbose = True, tol=1e-30)
		model_early.fit(X) ## L, n by k
		F = model_early.components_.T
		S = []

if method == "random":
		model_early = NMF(n_components=k, init='random', random_state=0,
		            beta_loss='kullback-leibler', solver='mu', max_iter=1, verbose = True, tol=1e-30)
		model_early.fit(X) ## L, n by k
		F = model_early.components_.T
		S = []

W, H, _ = non_negative_factorization(X, n_components=k, init='custom', random_state=0,
                                          update_H=False, H=F.T,
                                          beta_loss='kullback-leibler', solver='mu',
                                          max_iter = 100, tol=1e-30, verbose = True)

loss = _beta_divergence(X, W, H, 1, square_root=True)
out = {'W': W, 'H': H, 'loss':loss, 'S': S}

file = open(outputfile, 'wb')
pickle.dump(out, file)
file.close()




