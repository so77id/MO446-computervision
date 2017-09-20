import numpy as np

def _manhattan(a, b):
	return np.sum(np.abs(a - b), axis = 1)

def _euclidean(a, b):
	return np.sqrt(np.sum(np.power(a - b, 2), axis = 1))

def minkowski(a, b, p = 2.0):
	
	if p == 1.0:
		return _manhattan(a, b)
	
	if p == 2.0:
		return _euclidean(a, b)

	return np.power(np.sum(np.power(np.abs(a - b), p), axis = 1), 1.0 / p)