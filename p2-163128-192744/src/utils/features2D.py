from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import cv2
import numpy as np
import scipy.stats as st

def flatten(array):
	return [item for sublist in array for item in sublist]


def gaussian_kernel(kernlen=5, depth=8, nsig=3):

    interval = (2*nsig+1.)/(kernlen)
    x = np.linspace(-nsig-interval/2., nsig+interval/2., kernlen+1)
    kern1d = np.diff(st.norm.cdf(x))
    kernel_raw = np.sqrt(np.outer(kern1d, kern1d))
    kernel = kernel_raw/kernel_raw.sum()

    final_kernel = np.zeros((kernlen,kernlen,depth))

    for i in range(depth):
    	final_kernel[:,:,i] = kernel

    return final_kernel

def kp_to_nparr(kps, height):

	x = np.zeros((len(kps), 2), dtype = np.float64)

	for i in range(len(kps)):
		x[i] = (kps[i].pt[1], kps[i].pt[0])

	return x

def is_local_opt(cube):
	min = True
	max = True
	for i in range(cube.shape[0]):
	    for j in range(cube.shape[1]):
	        for k in range(cube.shape[2]):
	            if i == 1 and j == 1 and k == 1:
	                continue
	            if max and cube[i,j,k] >= cube[1,1,1]:
	                max = False
	                if not min:
	                    return False
	            if min and cube[i,j,k] <= cube[1,1,1]:
	                min = False
	                if not max:
	                    return False
	return True


def find_local_opt(dog, ratio, cthreshold):
	local_opt_dog = []

	for octave in dog:

		oct_kp = []

		for s in range(1, octave.shape[0]-1):
			scale = octave[s]
			kp = []

			for i in range(1, scale.shape[0]-1):
				for j in range(1, scale.shape[1]-1):
					if is_local_opt(octave[s-1:s+2,i-1:i+2, j-1:j+2]):
						# print("Candidato")
						dx = (octave[s, i+1, j] - octave[s, i-1, j]) * 0.5 / 255
						dy = (octave[s, i, j+1] - octave[s, i, j-1]) * 0.5 / 255
						ds = (octave[s+1, i, j] - octave[s-1, i, j]) * 0.5 / 255

						dxx = (octave[s, i+1, j] + octave[s, i-1, j] - 2 * octave[s, i, j]) / 255
						dyy = (octave[s, i, j+1] + octave[s, i, j-1] - 2 * octave[s, i, j]) / 255
						dss = (octave[s+1, i, j] + octave[s-1, i, j] - 2 * octave[s, i, j]) / 255

						dxy = (octave[s, i-1, j-1] + octave[s, i+1, j+1] - octave[s, i-1, j+1] - octave[s, i+1, j-1]) * 0.25 / 255
						dxs = (octave[s-1, i-1, j] + octave[s+1, i+1, j] - octave[s-1, i+1, j] - octave[s+1, i-1, j]) * 0.25 / 255
						dys = (octave[s-1, i, j-1] + octave[s+1, i, j+1] - octave[s-1, i, j+1] - octave[s+1, i, j-1]) * 0.25 / 255

						dD = np.matrix([[dx] , [dy] , [ds]])
						H = np.matrix([[dxx, dxy, dxs], [dxy, dyy, dys], [dxs, dys, dss]])

						x_hat = np.linalg.inv(H.transpose() * H) * (H.transpose() * dD)

						d_x_hat = octave[s,i,j] + np.dot(dD.transpose(), x_hat) * 0.5

						if (dxx + dyy) **2 * ratio < ((dxx*dyy - dxy**2) * (ratio+1)**2) and \
						    np.absolute(d_x_hat) >= cthreshold and \
						    np.prod(np.absolute(x_hat) < 0.5):
						    kp.append([i,j])
						    #print("Key point:", [i,j])

			# print("s:", s, "kp:", kp)
			oct_kp.append(kp)

		#print("octave:", oct_kp)
		local_opt_dog.append(oct_kp)

	#print("Final:", local_opt_dog)
	return local_opt_dog


def find_kp_features(img, kps, nbins, kernel, factor):

	dx = np.zeros(img.shape, dtype = img.dtype)
	dy = np.zeros(img.shape, dtype = img.dtype)

	dx[:,1:-1] = img[:,:-2] - img[:,2:]
	dy[1:-1,:] = img[:-2,:] - img[2:,:]

	mag = np.sqrt(dy ** 2 + dx ** 2)
	ori = np.arctan(dy / dx)


	inds = np.array(range(8))
	half_kernel = [x/2 for x in kernel]
	gaussian_cube = gaussian_kernel(kernlen=4, depth=8, nsig=2)

	f_kps = []
	f_descs = []

	for kp in kps:
		i, j = kp
		li = int(i - half_kernel[0])
		ui = int(i - half_kernel[0] + 1)

		lj = int(j - half_kernel[1])
		uj = int(j - half_kernel[1] + 1)

		if li < 0 or ui >= img.shape[0] or lj < 0 or uj >= img.shape[1]:
			continue

		mag_roi = mag[li:ui, lj:uj]
		ori_roi = ori[li:ui, lj:uj]

		hist, ranges =  np.histogram(ori_roi, bins=nbins, range=(-np.pi, np.pi), weights=mag_roi)
		max_i = np.argmax(hist)
		max_v = hist[max_i] * 0.8
		if max_v == 0:
			continue

		ori_candidates = np.where(hist >= max_v)[0]

		for ori_c in ori_candidates:
			#c_ori = (ranges[ori_c] + ranges[ori_c+1])/2

			if i < 7 or j < 7 or i > img.shape[0] - 7 or j > img.shape[0] - 7:
				continue

			i_ = i - 8
			j_ = j - 8

			desc_cube = np.zeros((4,4,8), dtype=np.float64)
			for k in range(4):
				for l in range(4):
					desc_hist, desc_ranges = np.histogram(ori[ i_+4*k : i_+4*k+4 ,  j_+4*l : j_+4*l+4  ], bins=8, range=(-np.pi, np.pi), weights=mag[ i_+4*k : i_+4*k+4 ,  j_+4*l : j_+4*l+4  ])
					# Rotation invariance
					desc_hist = desc_hist[(inds-ori_c) % 8]

					# Ilumination invariance
					sum = desc_hist.sum()
					if sum != 0:
						desc_hist = desc_hist / desc_hist.sum()
						desc_hist[desc_hist > 0.2] = 0.2
						desc_hist /= desc_hist.sum()

					desc_cube[k,l] = desc_hist

			desc_cube = desc_cube * gaussian_cube
			desc = desc_cube.flatten()

			f_kps.append(np.array([ (float(i)/factor), (float(j)/factor) ]))
			f_descs.append(np.array(desc))

	return f_kps, f_descs

def CVSIFT(img):
	sift = cv2.xfeatures2d.SIFT_create()

	key_points, descriptors   = sift.detectAndCompute(img, None)

	return kp_to_nparr(key_points, img.shape[0]), descriptors


def SIFT(img):
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	return CVSIFT(gray)
	#return OUR_SIFT(gray)



def OUR_SIFT(img, k = 2 ** 0.5, init_sigma = 1.0 / (2 ** 0.5), kernel = (5,5), scales = 5, octaves = 4, cthreshold = 0.03, ratio=10, nbins = 36):
	assert(scales >= 5)

	kps = []
	des = []

	factors = [2] # It is recommended that the first octave takes the image doubled in size
	for i in range(1, octaves):
		factors.append(factors[-1] / 2.0)

	scale_space = []
	for i in range(len(factors)):

		img_resized = cv2.resize(img, None, fx = factors[i], fy = factors[i])
		octave = np.zeros((scales, int(img_resized.shape[0]), int(img_resized.shape[1])))

		for scale in range(scales):
			octave[scale] = cv2.GaussianBlur(img_resized, kernel, init_sigma * (k ** (2 * i + scale)))

		scale_space.append(octave)

	dog = []
	for octave in scale_space:
		dog.append(np.diff(octave, axis = 0))

	local_opt_dog = find_local_opt(dog,ratio, cthreshold)

	f_kps = []
	f_desc = []
	for o in range(len(local_opt_dog)):
		kp_or = []
		for s in range(len(local_opt_dog[o])):
			img = scale_space[o][s]
			kps = local_opt_dog[o][s]
			kps, descs  = find_kp_features(img, kps, nbins, kernel, factors[o])
			f_kps.append(kps)
			f_desc.append(descs)

	return np.array(flatten(f_kps)), np.array(flatten(f_desc))

if __name__ == '__main__':
	img = cv2.cvtColor(cv2.imread("/home/juan/git/MO446-computervision/p2-163128-192744/output/original/_frame1.png"), cv2.COLOR_BGR2GRAY)
	print(img.shape)

	kps, desc = SIFT_kps_fts(img)
	print(np.unique(kps, axis=0).shape)
	print(desc.shape)


