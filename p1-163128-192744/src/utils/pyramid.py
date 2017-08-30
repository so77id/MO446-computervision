from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import cv2
import numpy as np

import utils.scale
import utils.conv

def PyUp(image, mask):
        up_level = cv2.filter2D(image,-1, mask)
        up_level = utils.scale.downsamplingX2(up_level)
        return up_level

def PyDown(image, shape):
    return cv2.resize(image, shape[::-1], interpolation=cv2.INTER_LINEAR)
    # return utils.scale.bilinearX2(image, shape)

class Pyramid:
    """
        Pyramid abstract class, superclass of GaussianPyramid and
        LaplacianPyramid.

        Attributes:

        img     : Input image
        P       : Pyramid obtained from successive filtering of img. Its
                  element at index 0 is img.
        l       : Number of levels, besides the original image
    """
    def __init__(self, img, l):

        self.img = img
        self.l = l
        self.__P = []

    def append_level(self, new_level):
        self.__P.append(new_level)

    def push_level(self, new_level):
        self.__P = [new_level] + self.__P

    def del_level(self, index):
        del self.__P[index]

    def access(self, i):
        return self.__P[i]

    def range(self):
        return range(self.l + 1)

    def inv_range(self):
        return range(self.l, -1, -1)

    def composition(self):
        # height, width, channels = img.shape
        shape = self.__P[0].shape
        new_shape = list(shape)
        new_shape[1] = int(new_shape[1] + new_shape[1]/2)
        new_shape = tuple(new_shape)
        py_img = np.zeros(new_shape, dtype=self.__P[0].dtype)

        py_img[0:shape[0],0:shape[1]] = self.__P[0]
        prev_shape = 0
        for i in range(1, self.l + 1, 1):
            py_img[prev_shape:(prev_shape + self.__P[i].shape[0]), (shape[1]):(shape[1] + self.__P[i].shape[1])] = self.__P[i]
            prev_shape += self.__P[i].shape[0]

        return py_img



class GaussianPyramid(Pyramid):
    """
        GaussianPyramid class, inherits from Pyramid

        Attributes:
            mask    : Filter to be applied successively
    """

    def __init__(self, img, l, mask):
        Pyramid.__init__(self, img, l)
        self.mask = mask

        self.append_level(img)
        for i in range(self.l):
            self.append_level( PyUp(self.access(-1), self.mask) )

class LaplacianPyramid(Pyramid):
    """
        LaplacianPyramid class, inherits from Pyramid

        Attributes:
            GP    : Gaussian Pyramid to apply filtering
    """

    def __init__(self, GP):
        self.GP = GP
        Pyramid.__init__(self, GP.img, GP.l)

        self.push_level(self.GP.access(-1))
        for i in range(self.l, 0, -1):
            self.push_level(cv2.subtract(self.GP.access(i-1), PyDown(self.GP.access(i), self.GP.access(i-1).shape)))
