from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import cv2
import numpy as np
from numpy.linalg import lstsq

def optical_flow(frame0, frame1, p0, mode="cv2", neigh = 15, s0 = None):
	if mode is "cv2":
		p1, st = get_cv2_flow(frame0, frame1, p0)

	elif mode is "own":
		p1, st = get_flow(frame0, frame1, p0.reshape((-1,2)), neigh, s0)
		p1 = p1.reshape((-1,1,2))
		st = st.reshape((-1,1))

	return p1, st

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

		x, y = p0[p].astype(np.int16)
		if  s0[p] == 0 or x+w_ran[0] < 0 or y+w_ran[0] < 0 \
		 	or x+w_ran[-1] >= frame1.shape[1]-1 \
			or y+w_ran[-1] >= frame1.shape[0]-1:

			st[p] = 0
			p1[p] = [-1.,-1.]
			continue

		A = np.zeros((neigh ** 2, 2), dtype=np.float32)
		b = np.zeros((neigh ** 2), dtype=np.float32)

		k = 0
		for wx in w_ran:
			for wy in w_ran:
				A[k,0] = dx[y + wy, x + wx]
				A[k,1] = dy[y + wy, x + wx]
				b[k]   = dt[y + wy, x + wx]
				k += 1

		p1[p] = p0[p] + lstsq(A, -b)[0]
		st[p] = 0 <= p1[p,0] < frame1.shape[1] - 1 \
				and 0 <= p1[p,1] < frame1.shape[0] - 1

	return p1, st


def get_cv2_flow(frame0, frame1, p0):
	#Parameters for lucas kanade optical flow
	lk_params = dict( winSize  = (15,15),
	                  maxLevel = 2,
	                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

	p1, st, err = cv2.calcOpticalFlowPyrLK(frame0, frame1, p0, None, **lk_params)

	return p1, st
