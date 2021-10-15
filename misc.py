import numpy as np
import rpy2.robjects as robjects
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA




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
def vis_extremal_pca(X, S, which_dim = [0, 1]):
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
	# plt.legend(loc='upper left');
	plt.show()

def Cbar_proj(C):
	C = C/C.sum(axis=1)[:,None]
	pca = PCA(n_components=10)
	pca.fit(C)
	
	return pca.transform(C)