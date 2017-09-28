from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import cv2
import numpy as np
import key_points
from numpy.linalg import lstsq

def get_motion_map(V, neigh = 15):

	"""
	Recieves a video represented by an np.array V[t][h][w]
	
	It does not work still!

	dt = np.diff(V, axis = 0)
	dx = np.diff(V, axis = 2)
	dy = np.diff(V, axis = 1)

	w_ran = range(-(neigh // 2), (neigh // 2) + (neigh % 2) )

	for f in range(V.shape[0] - 1):

		img = V[f]
		X = key_points.decect_corners_harris(img)

		A = np.zeros((neigh ** 2, 2))
		b = np.zeros(neigh ** 2)

		for p in X:

			i = p[0]
			j = p[1]

			k = 0
			for wi in w_ran:
				for wj in w_ran:
					A[k,0] =  dx[i + wi, j + wj]
					A[k,1] =  dy[i + wi, j + wj]
					b[k]   = -dt[i + wi, j + wj]
					k++
			

		h = lstsq(A, b)

"""

