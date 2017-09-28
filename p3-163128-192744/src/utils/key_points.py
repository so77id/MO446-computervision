from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import cv2
import numpy as np

def decect_corners_harris(img, blocksize = 2, ksize = 3, k = 0.04, th = 0.1, margin = 5):

	"""
	Returns a list of  image coordinates whose Harris-detector score
	is above th-percent of the maximum score of the image. The output
	is prunned, so all the poins lie inside the specified margin.
	"""

	dst = cv2.cornerHarris(img, blocksize, ksize, k)
	iind, jind = np.where(dst > th * dst.max())

	sel = (iind >= margin) * (jind >= margin) * (iind < dst.shape[0] - margin) * (jind < dst.shape[1] - margin)

	iind = np.reshape(iind[sel], (np.sum(sel),1))
	jind = np.reshape(jind[sel], (np.sum(sel),1))

	return np.concatenate((iind, jind), axis = 1)

