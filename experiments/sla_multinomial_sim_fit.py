import os
import sys
import pickle
import argparse
from scipy import sparse
import numpy as np

script_dir = "../"
sys.path.append(os.path.abspath(script_dir))
from file2 import *
from factorize import *
from smallsim_functions import simulate_multinomial_counts
from misc import read_fitted_rds


parser = argparse.ArgumentParser(description='sla_multinomial_sim_fit')
parser.add_argument('-d', type=str)
parser.add_argument('-o', type=str)
parser.add_argument('-r', type=float)

args = parser.parse_args()

datafile=args.d
outputfile=args.o
rate=args.r


print(datafile)
print(outputfile)
print(rate)

np.random.seed(123)

# rate=1
# datafile="../../ebpmf_data_analysis/output/fastTopics_fit/fit_sla_fastTopics_k6.Rds"
# outputfile="output/fit_sim_sla_fastTopics_k6_rate1.pkl"

print("load model")
fitted = read_fitted_rds(datafile)
F = fitted['F']
L = fitted['L']
s = rate*fitted['s'].astype(int)

print("simulate data")
X,w_idct, F = simulate_multinomial_counts(L, F, s)
X = sparse.coo_matrix(X)

print("get Ctrue, Atrue")
n, p = X.shape
k = F.shape[1]
Ctrue = F @ L.T
Ctrue = Ctrue @ Ctrue.T
Ctrue /= Ctrue.sum()
Atrue = (L.T @ L) / n

print("fit on data")
C, _, _ = X2C(X)
S, B, A, _, _, _, _, C = factorizeC(C, K=k, rectifier='AP', optimizer='activeSet')

print("get S0 from Ctrue")
S0, B0, A0, _, _, _, _, _ = factorizeC(Ctrue, K=k, rectifier='no', optimizer='activeSet')

print("fit using S0")
# Perform row-normalization for the (rectified) co-occurrence matrix C.
C_rowSums = C.sum(axis=1)
# pdb.set_trace()
Cbar = C/C_rowSums[:,None]
# Step 2: Recover object-cluster matrix B. (i.e., recovers word-topic matrix)
print("+ Start recovering the object-cluster B...")
B2, _, _ = recoverB(Cbar, C_rowSums, S0, 'activeSet')
# Step 3: Recover cluster-cluster matrix A. (i.e., recovers topic-topic matrix)
print("+ Start recovering the cluster-cluster A...")
A2, _ = recoverA(C, B2, S0)


out = {'X':X, 'Ftrue':F, 'Ltrue':L, 'Atrue':Atrue, 
'C':C, 'S':S, 'B':B, 'A':A, 
'S0':S0, 'B0':B0, 'A0':A0,
'B2':B2, 'A2':A2}

file = open(outputfile, 'wb')
pickle.dump(out, file)
file.close()
