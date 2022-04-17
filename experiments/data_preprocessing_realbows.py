import os
import sys
# import matplotlib.pyplot as plt

import numpy as np
from scipy import sparse, io

# import pickle
# from sklearn.decomposition import NMF, non_negative_factorization
# from sklearn.decomposition._nmf import _beta_divergence

script_dir = "../"
sys.path.append(os.path.abspath(script_dir))
from file3 import *
# from factorize import *
# from misc import *

np.random.seed(123)

# C, D1, D2, dictionary = createC("../dataset/real_bows/docword.nips.txt", "../dataset/real_bows/vocab.nips.txt", "../dataset/real_bows/standard.stops", N=5000, min_objects=3, min_tokens=5, output_filename="example")

dir_name="../dataset/real_bows"
data_name="nips"
vocab_size = 4000
min_objects=10
min_tokens=10

# data_name="nytimes"
# vocab_size = 15000
# min_objects=3
# min_tokens=5

docword_file=f"{dir_name}/docword.{data_name}.txt"
vocab_file=f"{dir_name}/vocab.{data_name}.txt"
stop_file=f"{dir_name}/standard.stops"


bows, dictionary = readBows(docword_file, vocab_file, stop_file, 
	N=vocab_size, min_objects=min_objects, output_filename="")

#pdb.set_trace()

# ## remove words that only appear in one document
# p = dictionary.shape[0]
# idx = []
# w_idx = []
# for w in range(p):
# 		if (bows[:, 1]-1 == w).sum() < 2:
# 				idx += [np.where(bows[:, 1]-1 == w)[0][0]]
# 				w_idx += [w]
# bows = np.delete(bows, np.array(idx), axis = 0)
# dictionary = np.delete(dictionary, np.array(w_idx))


## get X and C
X, _, _ = bows2H(bows, min_tokens=min_tokens)

print((X > 0).sum(axis = 0).min())

C, _, _ = bows2C(bows, min_tokens=min_tokens)

io.mmwrite(f"{dir_name}/{data_name}_X.mtx", X.T)
np.savetxt(f"{dir_name}/{data_name}_C.csv", C, delimiter=',')
dict_file = open(f"{dir_name}/{data_name}_dict.txt", "w")
for element in dictionary:
    dict_file.write(element + "\n")
dict_file.close()