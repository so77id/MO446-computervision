from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math

import cv2
import numpy as np

from scipy import spatial
# Theshold must grater than 30 because 27 es the mean of the mean value of distances between the best matches


def kp_matcher(des0, des1, threshold):
    tree = spatial.KDTree(des0)
    _, matches = tree.query(des1, distance_upper_bound = threshold)

    return matches