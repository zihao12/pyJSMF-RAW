import numpy as np
import rpy2.robjects as robjects

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
