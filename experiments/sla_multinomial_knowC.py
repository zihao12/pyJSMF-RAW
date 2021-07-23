import os
import sys
import pandas as pd
from scipy import sparse
from sklearn.decomposition import NMF, LatentDirichletAllocation

import numpy as np
import matplotlib.pyplot as plt
from scipy.io import mmread
script_dir = "../"
sys.path.append(os.path.abspath(script_dir))
from file2 import *
from factorize import *
from smallsim_functions_anchor import *
from misc import *


np.random.seed(123)

k = 20
datafile=f"../dataset/sla_simulated/sla_simulated_fastTopics_k{k}_1.mtx"
Lfile=f"../dataset/sla_simulated/sla_simulated_fastTopics_k{k}_1_trueL.csv"
Ffile=f"../dataset/sla_simulated/sla_simulated_fastTopics_k{k}_1_trueF.csv"

L = np.genfromtxt(Lfile)
F = np.genfromtxt(Ffile)
# X = mmread(datafile)


Ctrue = F @ L.T
Ctrue = Ctrue @ Ctrue.T
Ctrue = Ctrue/Ctrue.sum()

S0, B0, A0, Btilde0, _, _, _, _ = factorizeC(Ctrue, k, rectifier='no', optimizer="activeSet")

tpx_idx0 = match_topics(F, B0).astype(int)
print("matched topics")
print(tpx_idx0)

print("F[S0,] after permutation")
print(F[S0[tpx_idx0],].round(3))

print("B0[S0,] after permutation")
print(B0[np.ix_(S0[tpx_idx0],tpx_idx0)].round(3))


Atrue = (L.T @ L)/L.shape[0]
print("Atrue after permutation")
print(Atrue.round(3))

print("A0 after permutation")
print(A0[np.ix_(tpx_idx0, tpx_idx0)].round(3))