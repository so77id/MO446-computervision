from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import cv2
import numpy as np

def list_to_keypoints_hc(kps):
	new_kps = []
	for kp in kps:
		new_kps.append(cv2.KeyPoint(x=kp[1], y=kp[0], _size=1))

	return new_kps

def list_to_keypoints_gftt(kps):
	new_kps = []
	for kp in kps:
		new_kps.append(cv2.KeyPoint(x=kp[0][0], y=kp[0][1], _size=1))

	return new_kps

def list_to_matrix(kps):
	return np.array([ [[k.pt[0], k.pt[1]] ]for k in kps], dtype=np.float32)


def trim_borders(kps, shape, margin = 15):
	final_kps = []
	for kp in kps:
		if ((shape[1] - margin) >= kp.pt[0] >= margin) and ((shape[0] - margin) >= kp.pt[1] >= margin):
			final_kps.append(kp)

	return final_kps

def harris_corner(img, blocksize = 2, ksize = 3, k = 0.04, th = 0.1):

	"""
	Returns a list of  image coordinates whose Harris-detector score
	is above th-percent of the maximum score of the image. The output
	is prunned, so all the poins lie inside the specified margin.
	"""

	dst = cv2.cornerHarris(img, blocksize, ksize, k)
	iind, jind = np.where(dst > th * dst.max())

	iind = np.reshape(iind, (iind.shape[0],1))
	jind = np.reshape(jind, (jind.shape[0],1))


	kps = np.concatenate((iind, jind), axis = 1)
	return list_to_keypoints_hc(kps)

def sift(img):
	sift = cv2.xfeatures2d.SIFT_create()
	key_points = sift.detect(img)
	return key_points

def orb(img):
	orb = cv2.ORB_create()
	key_points = orb.detect(img,None)
	return key_points

def gftt(img):
	feature_params = dict( maxCorners = 100,
	                       qualityLevel = 0.6,
	                       minDistance = 2,
	                       blockSize = 3 )
	key_points = cv2.goodFeaturesToTrack(img, mask = None, **feature_params)

	return list_to_keypoints_gftt(key_points)

def detect_points(img, margin = 15, mode="sift"):

	if mode == "hc":
		kps = harris_corner(img)
	elif mode == "gftt":
		kps = gftt(img)
	elif mode == "sift":
		kps = sift(img)
	elif mode == "orb":
		kps = orb(img)

	kps_list = trim_borders(kps, img.shape, margin)
	kps_matrix = list_to_matrix(kps_list)
	return kps_list, kps_matrix
