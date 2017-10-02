from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import cv2
import numpy as np
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
		st[p] = 0 <= p1[p,0] < frame1.shape[0] - 1 \
				and 0 <= p1[p,1] < frame1.shape[1] - 1

	return p1, st

def get_motion_map(V, neigh = 15):
	pass
	"""
	Recieves a video represented by an np.array V[t][h][w]


	dt = np.diff(V, axis = 0)
	dx = np.diff(V, axis = 2)
	dy = np.diff(V, axis = 1)

	w_ran = range(-(neigh // 2), (neigh // 2) + (neigh % 2) )
	X = key_points.decect_corners_harris(V[0])

	for f in range(1, V.shape[0]):

		A = np.zeros((neigh ** 2, 2))
		b = np.zeros(neigh ** 2)

		for p in X:

			x = p[0]
			y = p[1]

			k = 0
			for wi in w_ran:
				for wj in w_ran:

	h = lstsq(A, b)
"""

if __name__ == '__main__':

	feature_params = dict( maxCorners = 100,
                       qualityLevel = 0.3,
                       minDistance = 7,
                       blockSize = 7 )

	lk_params = dict(winSize  = (15,15), maxLevel = 2,
                     criteria = (cv2.TERM_CRITERIA_EPS | \
					             cv2.TERM_CRITERIA_COUNT, 10, 0.03))


	frame0 =  cv2.cvtColor(cv2.imread('/home/juan/git/MO446-computervision/' + \
				'p0-163128-192744/input/p0-1-0.png'), cv2.COLOR_BGR2GRAY)
	frame1 = np.zeros(frame0.shape, dtype = frame0.dtype)
	frame1[:-1,:-1] = frame0[1:,1:]
	frame1 = np.transpose(frame0).reshape(frame0.shape)
	#frame1 = frame0
	p0 = cv2.goodFeaturesToTrack(frame0, mask = None, **feature_params)
	p0 = p0.reshape((p0.shape[0], 2))

	p1, st = get_flow(frame0, frame1, p0)
	p1_, st_, _ = cv2.calcOpticalFlowPyrLK(frame0, frame1, p0, None, **lk_params)

	print(p0[st==1])
	print(p1[st==1])
	print(p1_[st==1].astype(np.int16))
	#print(p1_[st==1] - p1[st==1])
