from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import cv2
import numpy as np
from skimage.feature import greycomatrix, greycoprops, local_binary_pattern
import scipy.stats as st

def get_mask_img(img, mask):
	ch1, ch2, ch3 = cv2.split(img)
	ch1 *= mask
	ch2 *= mask
	ch3 *= mask

	return [ch1, ch2, ch3]

def size(mask):
	return mask.sum()

def mean_color(img, mask):

	return img[mask == 1].mean(axis=0)

def GLCM_features(img, mask, bbox):
	x, y, w, h = bbox

	w = max(w, 1)
	h = max(h, 1)

	glcm_img = []
	for ch in get_mask_img(img, mask):
		glcm = greycomatrix(ch[y : y + h, x : x + w], [1], [0, np.pi / 4, np.pi/2])
		glcm = glcm[1:,1:]
		glcm_img.append(glcm)

	return glcm_img

def contrast(glcms):
	return [ greycoprops(glcm, 'contrast').tolist() for glcm in glcms]

def energy(glcms):
	return [ greycoprops(glcm, 'energy').tolist() for glcm in glcms]

def correlation(glcms):
	return [ delete_nan(greycoprops(glcm, 'correlation'), 1).tolist() for glcm in glcms]

def entropy(img, mask):
	return [ st.entropy(ch.reshape(-1)) for ch in get_mask_img(img, mask)]

def centroid_color(img, mask):
	return [0, 0]

def bounding_box(mask):
	ys, xs = np.where(mask == 1)

	x_min = xs.min()
	x_max = xs.max()

	y_min = ys.min()
	y_max = ys.max()

	w = x_max - x_min
	h = y_max - y_min

	return [x_min, y_min, w, h]

#https://stackoverflow.com/questions/12472338/flattening-a-list-recursively
def flatten(S):
    if S == []:
        return S
    if isinstance(S[0], list):
        return flatten(S[0]) + flatten(S[1:])
    return S[:1] + flatten(S[1:])

def delete_nan(descriptor, value):
	descriptor[np.isnan(descriptor)] = value
	return descriptor

def classic(img, mask):

	mask = mask.astype(np.uint8)
	descriptor = []

	bbox = bounding_box(mask)


	descriptor.extend(bbox[2:])

	descriptor.append(size(mask))
	descriptor.extend(mean_color(img, mask))

	glcm = GLCM_features(img, mask, bbox)
	descriptor.extend(contrast(glcm))
	descriptor.extend(correlation(glcm))


	descriptor.extend(entropy(img, mask))

	#Flatten
	descriptor = flatten(descriptor)

	return descriptor

def lbp(img, mask):

	x, y, w, h = bounding_box(mask)

	w = max(1, w)
	h = max(1, h)
	
	img = img[y : y + h, x : x + w]
	mask = mask[y : y + h, x : x + w]

	descriptor = []

	for ch in cv2.split(img):
		img_lbp = local_binary_pattern(ch, 8, 1.0)	
		descriptor.extend(np.histogram(img_lbp[mask.astype(np.bool)], bins=8, range=(0,8))[0].astype('float').tolist())

	return descriptor