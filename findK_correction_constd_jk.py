## implement corrected findK with jackknife
import numpy as np
import pdb

def findS_correction_jk(X, Cbar, C, K):
		n, p = X.shape
    d = X.sum() / n
    D = d * (d - 1)
    w = X.sum(axis = 0)
    w /= w.sum()

		Res = Cbar.T
		ind = np.zeros(K)
		diagR = np.zeros(K)

		g, v, S = JK_step1(X, C, w, D)
		g_ = shrink(g, v)

		for k in range(K):
				maxSquaredSum = np.max(g_)
				maxCol = np.argmax(g_)
				ind[k] = maxCol
				u = Res[:, maxCol].copy()
				u = u / np.sqrt(np.power(u, 2).sum()) ## u / np.sqrt(maxSquaredSum)? Could be better
				diagR[k] = np.sqrt(maxSquaredSum)
				
				g, v, S = JK_stepk(X, u, S, g, C, w, D)
				g_ = shrink(g, v)
				Res = Res -  u[:, None] @ (u[:,None].T @ Res)

		return S, diagR


def JK_step1(X, C, w, D):
		n, p = X.shape
		g = (X**2).sum(axis = 0) 
		X2 = np.power(X, 2)
		b = (X2.sum(axis = 1)[:, None] * X2).sum(axis = 0) + 	X2.sum(axis = 0) - 2 * mp.power(X, 3).sum(axis = 0)
		b /= (n - 1) * n
		g = (1 + 1/(n-1)) * g - b

		## compute S, v; 
		S = X2 * (1 + X2.sum(axis = 1)[:, None] - 2 * X) / np.power(D, 2)
		S -= S.mean(axis = 0)[None, :]
		S += 2 * n * np.power(C, 2).sum(axis = 1)
		S -= 2* (n/D) * X * (X @ C - np.diag(C)[None, :])
		S /= (n - 1) 
		v = np.power(S, 2).sum(axis = 0) / (n - 1)

		## scale g, v with w
		g /= np.power(w, 2)
		v /= np.power(w, 4)

		return g, v, S

		

def JK_stepk(X, u, S, g, C, w, D):
		utC = u[:, None].T @ C
		A = X * (X @ u)[:, None] - X * u[None, :]
		A /= D

		dg = np.power(utC, 2)
		dg = (1 + 1/(n-1)) * dg - np.power(A, 2).sum(axis = 0) / (n-1) / n
		g -= dg.T / np.power(w, 2)

		dS = np.power(A, 2)
		dS -= dS.mean(axis = 0)[None, :]
		dS -= 2 * n * (A - A.mean(axis = 0)[None, :]) * utC[None, :]
		dS /= (n-1)
		S -= dS
		v = np.power(S, 2).sum(axis = 0) / (n - 1)
		v /= np.power(w, 4)

		return g, v, S







def shrink(g, v):
		
		return g



