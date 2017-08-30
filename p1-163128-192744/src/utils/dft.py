from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import cv2
import numpy as np
from scipy.stats import rankdata

def dft_mp(img):
    dft = cv2.dft(np.float32(img), flags = cv2.DFT_COMPLEX_OUTPUT)
    return cv2.cartToPolar(dft[:,:,0], dft[:,:,1])

def idft_mp(dft_m, dft_p):
    return cv2.idft(cv2.merge(cv2.polarToCart(dft_m, dft_p)), flags=cv2.DFT_SCALE | cv2.DFT_REAL_OUTPUT)

def filter(dft_m, dft_p, threshold = None, dec = False, phase = True):

    if threshold is not None:
        assert(threshold >= 0.0 and threshold <= 1.0)

    npc = np.max if dec else np.min

    if phase:
        dft_msk = abs(dft_p - np.pi)
        target = dft_p
    else:
        dft_msk = dft_m
        target = dft_m

    if threshold is None:
        target = dft_p * (dft_msk == npc(dft_msk[np.nonzero(dft_msk)]))
    else:
        if dec:
            target = target * (np.resize(rankdata(dft_msk), dft_msk.shape) >= int(round(dft_msk.size * (1 - threshold))))
        else:
            target = target * (np.resize(rankdata(dft_msk), dft_msk.shape) <= int(round(dft_msk.size * threshold)))

    if phase:
        return (dft_m, target)
    else:
        return (target, dft_p)
