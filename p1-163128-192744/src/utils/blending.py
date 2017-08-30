from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import cv2

import scipy.ndimage.filters as fi
import utils.pyramid as pyramid
import utils.dft as dft
import numpy as np

def blend_img(img1, img2, mask):
    return img1 * (1 - mask) +  img2 * mask

def pyramid_blending(Py1, Py2, Pym, pyramid_size):
    blends = []
    for i in Pym.inv_range():
            mask = Pym.access(i)
            img1 = Py1.access(i)
            img2 = Py2.access(i)

            blend = blend_img(img1, img2, mask)

            blends.append(blend)

    blend = blends[0]
    for i in range(1, pyramid_size+1):
        blend = cv2.add(pyramid.PyDown(blend, blends[i].shape), blends[i])

    return blend

def blending(img1, img2, mask, pyramid_size, kernel):

    img1_GP = pyramid.GaussianPyramid(img1, pyramid_size, kernel)
    img2_GP = pyramid.GaussianPyramid(img2, pyramid_size, kernel)
    mask_GP = pyramid.GaussianPyramid(mask, pyramid_size, kernel)

    img1_LP = pyramid.LaplacianPyramid(img1_GP)
    img2_LP = pyramid.LaplacianPyramid(img2_GP)

    return pyramid_blending(img1_LP, img2_LP, mask_GP, pyramid_size)

def freq_pyramid_blending(img1, img2, mask, pyramid_size, kernel):
    img1_ = img1 * mask
    img2_ = img2 * (1 - mask)

    dft_m1, dft_p1 = dft.dft_mp(img1_)
    dft_m2, dft_p2 = dft.dft_mp(img2_)

    blend_m = blending(dft_m2, dft_m1, freq_mask, pyramid_size, kernel)
    blend_p = blending(dft_p2, dft_p1, freq_mask, pyramid_size, kernel)

    return dft.idft_mp(blend_m, blend_p)

def freq_concat_blending(img1, img2, mask):

    img1_ = img1 * mask
    img2_ = img2 * (1 - mask)

    dft_m1, dft_p1 = dft.dft_mp(img1_)
    dft_m2, dft_p2 = dft.dft_mp(img2_)

    dft_m1, dft_p1 = dft.filter(dft_m1, dft_p1, 0.05, True, False)
    dft_m2, dft_p2 = dft.filter(dft_m2, dft_p2, 0.05, True, False)

    blend_m = np.concatenate((dft_m1, dft_m2), axis = 0)
    blend_p = np.concatenate((dft_p1, dft_p2), axis = 0)

    return dft.idft_mp(blend_m, blend_p)[::2,:]