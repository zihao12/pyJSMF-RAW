import os
import sys
from scipy import sparse

import numpy as np
import pickle
from sklearn.decomposition import NMF, non_negative_factorization
from sklearn.decomposition._nmf import _beta_divergence
import magic
import scprep

script_dir = "../"
sys.path.append(os.path.abspath(script_dir))
from file2 import *
from factorize import *
from misc import *

datadir = "../../../gdrive/github_data/pyJSMF-RAW-data"
np.random.seed(123)

## load data
dataname = "sla"
k = 6
rate=0.5
outputfile=f"../experiments/anchor_word_initialization_sla_sim_rate{rate}.pkl"


datafile=f"{datadir}/fit_sim_{dataname}_fastTopics_k{k}_rate{rate}.pkl"
with open(datafile, "rb") as f:
    fitted = pickle.load(f)

X = fitted['X']
Ltrue = fitted['Ltrue']
Ftrue = fitted['Ftrue']
n, p = X.shape
out = {}



print("############## anchor word with true S0 #################")
Pi_true = Ltrue @ Ftrue.T
Ctrue = Pi_true.T @ Pi_true / n
Cbar0 = Ctrue / Ctrue.sum(axis = 1)[:, None]
S0, _, _ = findS(Cbar0, k)

Pi = X.toarray() / X.toarray().sum(axis = 1)[:, None]
C = (Pi.T @ Pi) / n
Cbar = C / C.sum(axis = 1)[:, None]
C_rowSums = C.sum(axis=1)

F, _, _ = recoverB(Cbar, C_rowSums, S0, "activeSet")
W, H, _ = non_negative_factorization(X, n_components=k, init='custom', random_state=0,
                                          update_H=False, H=F.T,
                                          beta_loss='kullback-leibler', solver='mu',
                                          max_iter = 100, verbose = True)

loss = _beta_divergence(X, W, H, 1, square_root=True)
out['W_anchor_oracle'] = W
out['H_anchor_oracle'] = H
out['loss_anchor_oracle'] = loss
print(loss)

print("############### loss at convergence ##################")
model = NMF(n_components=k, init='nndsvda', random_state=0,
            beta_loss='kullback-leibler', solver='mu', max_iter=100, verbose = True, tol=1e-30)
W = model.fit_transform(X) 
H = model.components_
loss = _beta_divergence(X, W, H, 1, square_root=True)
out['W0'] = W
out['H0'] = H
out['loss0'] = loss
print(loss)

## nndsvd
print("############### nndsvd ##################")
model_early = NMF(n_components=k, init='nndsvda', random_state=0,
            beta_loss='kullback-leibler', solver='mu', max_iter=1, verbose = True, tol=1e-30)
model_early.fit(X) ## L, n by k
H = model_early.components_
W, H, _ = non_negative_factorization(X, n_components=k, init='custom', random_state=0,
                                          update_H=False, H=H,
                                          beta_loss='kullback-leibler', solver='mu',
                                          max_iter = 100, verbose = True)

loss = _beta_divergence(X, W, H, 1, square_root=True)
out['W_nndsvd'] = W
out['H_nndsvd'] = H
out['loss_nndsvd'] = loss
print(loss)

## random init
print("############### random ##################")
model_early = NMF(n_components=k, init='random', random_state=0,
            beta_loss='kullback-leibler', solver='mu', max_iter=1, verbose = True, tol=1e-30)
model_early.fit(X) ## L, n by k
H = model_early.components_
W, H, _ = non_negative_factorization(X, n_components=k, init='custom', random_state=0,
                                          update_H=False, H=H,
                                          beta_loss='kullback-leibler', solver='mu',
                                          max_iter = 100, verbose = True)

loss = _beta_divergence(X, W, H, 1, square_root=True)
out['W_random'] = W
out['H_random'] = H
out['loss_random'] = loss
print(loss)

## magic
print("############### magic ####################")
Y_normalized = scprep.normalize.library_size_normalize(X.T) ## normalizing each gene; could amplify noise
Y_normalized = scprep.transform.sqrt(Y_normalized)
magic_op = magic.MAGIC()
magic_op.set_params(random_state = 12)
magic_op.fit(Y_normalized)

magic_op.set_params(t = 3)
Y_magic = magic_op.transform(X.T)

Cbar_m = Y_magic.T / Y_magic.T.sum(axis = 1)[:, None]
Cbar_m = Cbar_m.T @ Cbar_m / n
C_rowSums_m = Cbar_m.sum(axis = 1)
Cbar_m = Cbar_m / C_rowSums_m[:, None]
S_m, _, _ = findS(Cbar_m, k)

print(" ############ find F using naive Pi ################")
Pi = X.toarray() / X.toarray().sum(axis = 1)[:, None]
C = (Pi.T @ Pi) / n
Cbar = C / C.sum(axis = 1)[:, None]
C_rowSums = C.sum(axis=1)

F, _, _ = recoverB(Cbar, C_rowSums, S_m, "activeSet")
W, H, _ = non_negative_factorization(X, n_components=k, init='custom', random_state=0,
                                          update_H=False, H=F.T,
                                          beta_loss='kullback-leibler', solver='mu',
                                          max_iter = 100, verbose = True)

loss = _beta_divergence(X, W, H, 1, square_root=True)
out['W_magic0'] = W
out['H_magic0'] = H
out['loss_magic0'] = loss
print(loss)

print("############# find F using denoised Pi #################")
F, _, _ = recoverB(Cbar_m, C_rowSums_m, S_m, "activeSet")
W, H, _ = non_negative_factorization(X, n_components=k, init='custom', random_state=0,
                                          update_H=False, H=F.T,
                                          beta_loss='kullback-leibler', solver='mu',
                                          max_iter = 100, verbose = True)

loss = _beta_divergence(X, W, H, 1, square_root=True)
out['W_magic'] = W
out['H_magic'] = H
out['loss_magic'] = loss
print(loss)




print("############### vanila anchor word ####################")
C, D1, D2 = X2C(X)
print("############# no rectification ################")
S, B, A, Btilde, Cbar, C_rowSums, diagR, C = factorizeC(C, K=k, rectifier='no', optimizer='activeSet')
W, H, _ = non_negative_factorization(X, n_components=k, init='custom', random_state=0,
                                          update_H=False, H=B.T,
                                          beta_loss='kullback-leibler', solver='mu',
                                          max_iter = 100, verbose = True)

loss = _beta_divergence(X, W, H, 1, square_root=True)
out['W_anchor0'] = W
out['H_anchor0'] = H
out['loss_anchor0'] = loss
print(loss)


print("############# w/ rectification ################")
S, B, A, Btilde, Cbar, C_rowSums, diagR, C = factorizeC(C, K=k, rectifier='AP', optimizer='activeSet')
W, H, _ = non_negative_factorization(X, n_components=k, init='custom', random_state=0,
                                          update_H=False, H=B.T,
                                          beta_loss='kullback-leibler', solver='mu',
                                          max_iter = 100, verbose = True)

loss = _beta_divergence(X, W, H, 1, square_root=True)
out['W_anchor_ap'] = W
out['H_anchor_ap'] = H
out['loss_anchor_ap'] = loss
print(loss)


file = open(outputfile, 'wb')
pickle.dump(out, file)
file.close()

































