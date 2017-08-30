from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import cv2
import numpy as np

def bilinearX2(img, sh):

	assert(np.ceil(float(sh[0]) / 2) == img.shape[0])
	assert(np.ceil(float(sh[1]) / 2) == img.shape[1])

	img_ = np.zeros(sh, dtype = np.float32)

	for i in range(0, img_.shape[0], 2):
		for j in range(0, img_.shape[1], 2):
			img_[i, j] = img[int(i / 2), int(j / 2)]
			if i > 0:
				img_[i - 1, j] = (img_[i, j] + img_[i - 2, j]) / 2
			if j > 0:
				img_[i, j - 1] = (img_[i, j] + img_[i, j - 2]) / 2
			if i > 0 and j > 0:
					img_[i - 1, j - 1] = (img_[i, j - 1] + img_[i - 2, j - 1]) / 2


	if img_.shape[1] % 2 == 0:
		for i in range(img_.shape[0]):
			img_[i, img_.shape[1] - 1] = img_[i, img_.shape[1] - 2]

	if img_.shape[0] % 2 == 0:
		for i in range(img_.shape[1]):
			img_[img_.shape[0] - 1, i] = img_[img_.shape[0] - 2, i]

	return img_


def downsamplingX2(img):
	return img[::2,::2]


if __name__ == '__main__':

	img = cv2.imread('/home/juan/Documents/MO446/p1-1-a.png', 0)
	img_ = bilinearX2(img, (img.shape[0] * 2 - 1,img.shape[1] * 2 - 1))
	cv2.imwrite('/home/juan/Documents/MO446/p1-1-a_.png', img_)