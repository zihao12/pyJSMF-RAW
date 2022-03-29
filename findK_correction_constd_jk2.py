## implement corrected findK with jackknife
import numpy as np
import pdb

def findS_correction_jk(X, Cbar, C, K):
		n, p = X.shape
		d = X.sum() / n
		D = d * (d - 1)
		w = X.sum(axis = 0)
		w /= w.sum()
		W = compute_W(X, w, d)

		Res = Cbar.T
		S = np.zeros(K)
		diagR = np.zeros(K)

		theta, Theta = JK_step1(X, W, w, C, Cbar, D)
		g = theta
		G = Theta
		v = v_jk(G)
		g_ = shrink(g , v)

		for k in range(K):
				maxSquaredSum = np.max(g_)
				maxCol = np.argmax(g_)
				S[k] = maxCol
				u = Res[:, maxCol].copy()
				u = u / np.sqrt(np.power(u, 2).sum()) ## u / np.sqrt(maxSquaredSum)? Could be better
				diagR[k] = np.sqrt(maxSquaredSum) ## np.sqrt(np.power(u, 2).sum())?
				
				theta, Theta = JK_stepk(X, W, w, u, C, Cbar, D)
				g -= theta
				G -= Theta
				v = v_jk(G)
				g_ = shrink(g , v)

				Res = Res -  u[:, None] @ (u[:,None].T @ Res)

		return S, diagR

def compute_W(X, w, d):
		n = X.shape[0]
		out = (n-1) * d  * w[None, :] / (X.sum(axis = 0)[None, :] - X)

		return out

def JK_step1(X, W, w, C, Cbar, D): 
		n, p = X.shape
		X2 = np.power(X, 2)
		Theta = X2 * (1 + X2.sum(axis = 1)[:, None] - 2 * X) / np.power(D, 2)
		Theta -= 2* (n/D) * X * (X @ C - np.diag(C)[None, :])
		Theta /= -(n - 1) 
		Theta *= np.power(W, 2)
		Theta /= np.power(w, 2)[None, :] 
		Tn = (Cbar**2).sum(axis = 1) 
		Theta += n * (1 - (n / (n-1)) * np.power(W, 2)) * Tn[None, :]
		
		return Theta.mean(axis = 0), Theta

		

def JK_stepk(X, W, w, u, C, Cbar, D):
		n = X.shape[0]
		A = X * (X @ u)[:, None] - X * u[None, :]
		A /= D
		utCbar = Cbar @ u
		utC = C @ u
		Theta = np.power(A, 2)
		Theta -= 2 * n * A * utC[None, :]
		Theta /= -(n-1)
		Theta *= np.power(W, 2)
		Theta /= np.power(w, 2)[None, :] ## CHECK
		Tn = utCbar**2
		Theta += n * (1 - (n / (n-1)) * np.power(W, 2)) * Tn[None, :]
		
		return Theta.mean(axis = 0), Theta


def v_jk(G):
	n = G.shape[0]
	out = np.power(G - G.mean(axis = 0)[None, :], 2).mean(axis = 0) / (n-1)

	return out

 
def shrink(g, v):
		## careful not to alter g
		
		return g

## functions for experiments

def exper_jk(X, Cbar, C, U):
		K = U.shape[1]
		n, p = X.shape
		d = X.sum() / n
		D = d * (d - 1)
		w = X.sum(axis = 0)
		w /= w.sum()
		W = compute_W(X, w, d)

		gs = np.zeros((p, K))
		vs = np.zeros((p, K))

		Res = Cbar.T
		
		theta, Theta = JK_step1(X, W, w, C, Cbar, D)
		g = theta
		G = Theta
		v = v_jk(G)

		for k in range(K):
				gs[:, k] = g
				vs[:, k] = v
				u = U[:, k].copy()
				#pdb.set_trace()
				theta, Theta = JK_stepk(X, W, w, u, C, Cbar, D)
				g -= theta
				G -= Theta
				v = v_jk(G)
				
		return gs, vs

def get_U(Cbar, K):
		p = Cbar.shape[0]
		U = np.zeros((p, K))
		R = Cbar.T
		S = np.zeros(K)

		g = (R**2).sum(axis = 0) 
		for k in range(K):
				maxSquaredSum = np.max(g)
				maxCol = np.argmax(g)
				S[k] = maxCol
				u = R[:, maxCol].copy()
				u = u / np.sqrt(maxSquaredSum)
				U[:, k] = u
				R = R -  u[:, None] @ (u[:,None].T @ R)
				dg = (Cbar @ u)**2
				g -= dg        
				
		return S, U



