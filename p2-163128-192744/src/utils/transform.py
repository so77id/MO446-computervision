
import numpy as np
from scipy.interpolate import griddata

def transform(img, model, p):

	p_ = model.predict(p.astype(np.float64))
	img_ = np.zeros(img.shape, dtype = np.float64)

	for ch in range(img.shape[2]):
		grid = griddata(p_, img[:,:,ch].reshape((np.prod(img.shape[:2]))), p)
		grid[np.isnan(grid)] = 0
		for j in range(len(p)):
			img_[p[j,0],p[j,1],ch] = grid[j]

	return img_