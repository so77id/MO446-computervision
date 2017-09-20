
import numpy as np
from utils.error import minkowski

def ransac(x, y, model, k, t, d, l = 2.0, seed = None):
	"""
	Implementation of the original RANSAC method.

	Inputs:

		x     : Input
		y     : Target
		model : Model to be fitted
		k     : Number of iterations
		t     : Error threshold
		d     : Minimum number of candidate inliers
		l     : Minkowski p error parameter: 1.0 for manhattan, 2.0 for euclidean
		seed  : Seed of the random algorithm

	"""

	n = model.n_min

	if seed is not None:
		np.random.seed(seed)

	best = None
	e_min = np.inf
	inds = np.array(range(x.shape[0]))
	i = 0
	while(i < k):
		model_ = model.clone()
		np.random.shuffle(inds)

		try:
			model_.fit(x[inds[:n]], y[inds[:n]])
		except np.linalg.LinAlgError:
			continue

		y_ = model_.predict(x[inds[n:]])

		mkw = np.array(minkowski(y[inds[n:]], y_)).flatten()

		alsoinliers = np.where(mkw < t)[0]

		if alsoinliers.size <= d:
			i += 1
			continue

		alsoinliers = np.append(inds[:n], inds[alsoinliers + n])

		model_.fit(x[alsoinliers], y[alsoinliers])
		e = np.mean(minkowski(y[alsoinliers], model_.predict(x[alsoinliers]), l))

		if e < e_min:
			e_min = e
			best = model_

		i += 1

	return best, e_min

