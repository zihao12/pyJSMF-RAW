import numpy as np
import rpy2.robjects as robjects
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from scipy.stats import betabinom
from scipy.optimize import minimize





def read_fitted_rds(filename):
	readRDS = robjects.r['readRDS']
	data = readRDS(filename)
	d = {key: np.asarray(data.rx2(key)) for key in data.names}

	return d

def match_topics(F1, F2):
	k = F1.shape[1]
	ind = np.empty(k)
	for j1 in range(k):
		f1 = F1[:,j1]
		dist_min = np.inf
		for j2 in range(k):
			dist = np.sum((f1 - F2[:,j2])**2)
			if dist < dist_min:
				dist_min = dist
				matched = j2
		ind[j1] = matched

	return ind




## visualize the extremal rows of matrix X (intended for when X is a lower dimension embedding)
def vis_extremal_pca(X, S, which_dim = [0, 1], annotate=False,fontsize=6):
	mask = np.zeros(X.shape[0])
	mask[S] = 1
	mask = mask.astype(bool)

	fig = plt.figure()
	ax1 = fig.add_subplot(111)
	ax1.scatter(X[np.invert(mask),which_dim[0]], 
	            X[np.invert(mask), which_dim[1]], 
	            s = 3, c='b', marker="+", label='first')
	ax1.scatter(X[mask,which_dim[0]], 
	            X[mask, which_dim[1]], 
	            s=30, c='r', marker="o", label='second')
	if annotate:
		for s in S:
			ax1.annotate(s, (X[s,which_dim[0]], X[s,which_dim[1]]), 
				fontsize = fontsize, rotation=45)
	# plt.legend(loc='upper left');
	plt.show()

def vis_extremal2(X, S0, S, which_dim = [0, 1], annotate=False,fontsize=6):
	mask0 = np.zeros(X.shape[0])
	mask0[S0] = 1
	mask0 = mask0.astype(bool)

	mask = np.zeros(X.shape[0])
	mask[S] = 1
	mask = mask.astype(bool)

	fig = plt.figure()
	ax1 = fig.add_subplot(111)
	ax1.scatter(X[:,which_dim[0]], 
	            X[:, which_dim[1]], 
	            s = 3, c='b', marker="+", label='first')
	ax1.scatter(X[mask,which_dim[0]], 
	            X[mask, which_dim[1]], 
	            s=30, c='r', marker="o", label='second')
	ax1.scatter(X[mask0,which_dim[0]], 
	            X[mask0, which_dim[1]], 
	            s=30, c='green', marker="o", label='third')
	if annotate:
		for s in S:
			ax1.annotate(s, (X[s,which_dim[0]], X[s,which_dim[1]]), 
				fontsize = fontsize, rotation=45)
		for s in S0:
			ax1.annotate(s, (X[s,which_dim[0]], X[s,which_dim[1]]), 
				fontsize = fontsize, rotation=45)

	# plt.legend(loc='upper left');
	plt.show()



def Cbar_proj(C):
	C = C/C.sum(axis=1)[:,None]
	pca = PCA(n_components=10)
	pca.fit(C)
	
	return pca.transform(C)

def pca_proj(X):
	pca = PCA(n_components=10)
	
	
	return pca.fit_transform(X)

def betabinom_shrinkage(n, x, ab = np.array([1,1])):
    ab = mle_bb(n, x, ab)
    mu, var =  beta_mean_var(ab[0], ab[1])
    
    return pos_beta(n, x, ab), ab, mu, var

def obj_bb(ab_root, n, x):
    ab = ab_root**2
    return -betabinom.logpmf(x, n, ab[0], ab[1], loc=0).sum()

def mle_bb(n, x, ab):
    res = minimize(obj_bb, np.sqrt(ab), args=(n, x), method='BFGS')
    
    return res.x**2

def pos_beta(n, x, ab):
    
    return (ab[0] + x) / (ab.sum() + n)

def beta_mean_var(a, b):
    mu = a / (a + b)
    var = a*b/((a + b + 1) * (a+b)**2)
    
    return mu, var