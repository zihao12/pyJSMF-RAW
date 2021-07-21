import time
import numpy as np
from scipy import sparse
import pdb

##
# Main: bows2C()
#
# Inputs:
#   - bows: X-by-3 bag-of-words matrix (need to make sure all words have >= 1 tokens)
#   - min_tokens: min. number of tokens for effective training examples
#
# Outputs:
#   - C: NxN dense joint-stochastic co-occurrence matrix 
#   - D1: Nx1 example frequency where D1_i = # of examples where object i occurs
#   - D2: NxN co-example frequency where D2_{ij} = # of examples where object i and j co-occurs
#
# Remark:
#   - This function converts bag-of-words to the full/dense co-occurrence and
#     example/co-example frequencies by sequentially processing each document.
#
def bows2C(bows, min_tokens):
	#pdb.set_trace()
	# Print out the initial status
	print('[file.bows2C] Start constructing dense C...')

	# Recompute the size of vocabulary by counting the unique elements in the word numbers
	N = len(np.unique(bows[:,1]))
	M = bows[-1,0]

	# Find the row numbers where each training example ends
	endRows = np.where(bows[:-1,0] != bows[1:,0])[0]
	endRows += 1
	endRows = np.append([0], endRows)
	endRows = np.append(endRows, [len(bows)])

	# Compute co-occurrence and example/co-example frequencies for each training example.
	print('- Counting the co-occurrence for each document...')
	startTime = time.time()

	C = np.zeros((N, N))
	D1 = np.zeros(N)
	D2 = np.zeros((N,N))
	for m in range(len(endRows)-1):
		# print(m)

		# if m == 1:
		# 	pdb.set_trace()
		# Determine the start and end rows for this document
		startRow = endRows[m]
		endRow = endRows[m+1]
		objects = bows[startRow:endRow, 1]-1
		counts = bows[startRow:endRow, 2]

		# Skip the degenerate case when the document contains only one word with a single occurrence.
		# Note that it does not happen if min_object threshold is larger than 1 when reading bows.
		numObjects = len(objects)
		numTokens = sum(counts)
		if (numObjects == 1) and (numTokens == 1):
			continue

		# Skip the current example with less than minimum counts
		if numTokens < min_tokens:
			print('  - The document %d with only %d tokens will be ignored!' % (m, numTokens))
			continue

		# Accumulate corresponding counts to co-occurrence and example/co-example frequencies.
		# Note that co-example frequency for an object can exit only when the object occurs more than once.
		normalizer = numTokens*(numTokens-1)

		C[np.ix_(objects,objects)] += (np.outer(counts,counts) - np.diag(counts)).astype(float)/normalizer
		D1[objects] += 1
		D2[np.ix_(objects, objects)] += 1 - np.diag(counts == 1)

	# Ensure the overall sum is equal to 1.0
	entrySum = C.sum()
	# if (entrySum != M):
	# 	C /= entrySum
	C /= entrySum

	elapsedTime = time.time() - startTime

	# Print out the final status
	print('+ Finish constructing C and D!')
	print('  - The sum of all entries = %.6f' % (entrySum/M))
	print('  - Elapsed Time = %.4f seconds' % elapsedTime)

	return C, D1, D2

# Inputs:
#  - X: MxN document-word matrix, coo_matrix
# Outputs:
#  - bows: matrix of filtered examples where each row contains (example #, object #, frequency)
## NEED TO ADD 1 TO THE INDEX!!!
def X2Bows(X):
	return np.column_stack((X.row + 1, X.col + 1, X.data.astype(int)))

# Inputs:
#  - X: MxN document-word matrix, csc_matrix
# Outputs:
#  - C:  N-by-N full co-occurrence matrix
def X2C(X, min_tokens=1):
	return bows2C(X2Bows(X), min_tokens=min_tokens)


def createC(bows_filename, dict_filename, min_tokens =1):

	bows = readBows_from_mtx(bows_filename)
	dictionary = readObjects(dict_filename)
	C, D1, D2 = bows2C(bows, min_tokens)

	return C, D1, D2, dictionary

##
# Main: readBows_from_mtx()
  
def readBows_from_mtx(bows_filename):
	bows = np.genfromtxt(bows_filename, skip_header = 2, dtype='int')
	return bows

def readObjects(filename):

	# Open the file and read each object/word from every line
	file = open(filename, 'r')
	objects = np.genfromtxt(file, dtype='str')
	file.close()

	return objects
