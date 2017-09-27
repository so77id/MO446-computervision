from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import cv2
import numpy as np

def create_grid(img0, img1):
    assert(img0.shape == img1.shape)

    return np.concatenate((img0, img1), axis=1)