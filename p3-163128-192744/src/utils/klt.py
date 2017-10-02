from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import cv2
import numpy as np
import key_points
from numpy.linalg import lstsq

def get_flow(frame0, frame1, p0, neigh = 15, s0 = None):

	if s0 is None:
		s0 = np.ones(p0.shape[:1], dtype=np.int8)

	p1 = np.zeros(p0.shape, dtype=np.float32)
	st = np.ones(p0.shape[:1], dtype=np.int8)

	dt = frame1 - frame0
	dx = np.diff(frame0, axis = 1)
	dy = np.diff(frame0, axis = 0)

	w_ran = range(-(neigh // 2), (neigh // 2) + (neigh % 2) )

	for p in range(p0.shape[0]):

		i, j = p0[p].astype(np.int16)
		if  s0[p] == 0 or i+w_ran[0] < 0 or j+w_ran[0] < 0 \
		 	or i+w_ran[-1] >= frame1.shape[0]-1 \
			or j+w_ran[-1] >= frame1.shape[1]-1:

			st[p] = 0
			p1[p] = [-1.,-1.]
			continue

		A = np.zeros((neigh ** 2, 2), dtype=np.float32)
		b = np.zeros((neigh ** 2), dtype=np.float32)

		k = 0
		for wi in w_ran:
			for wj in w_ran:
				A[k,0] = dx[i + wi, j + wj]
				A[k,1] = dy[i + wi, j + wj]
				b[k]   = dt[i + wi, j + wj]
				k += 1

		p1[p] = lstsq(A, b)[0]
		st[p] = 0 <= p1[p,0] < frame1.shape[0] - 1 \
				and 0 <= p1[p,1] < frame1.shape[1] - 1

	return p1, st

def get_motion_map(V, neigh = 15):
	pass

