from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import scipy.stats as st
import cv2

#CONSTANTS
SAME_CONSTANT='SAME'
VALID_CONSTANT='VALID'


# GAUSSIAN KERNEL GENERATOR
# https://stackoverflow.com/questions/29731726/how-to-calculate-a-gaussian-kernel-matrix-efficiently-in-numpy
def gaussian_kernel(kernlen=5, nsig=3):
    """Returns a 2D Gaussian kernel array."""

    interval = (2*nsig+1.)/(kernlen)
    x = np.linspace(-nsig-interval/2., nsig+interval/2., kernlen+1)
    kern1d = np.diff(st.norm.cdf(x))
    kernel_raw = np.sqrt(np.outer(kern1d, kern1d))
    kernel = kernel_raw/kernel_raw.sum()

    return kernel


"""
Opencv3 border constants
    cv2.BORDER_CONSTANT - Adds a constant colored border. The value should be given as next argument.
    cv2.BORDER_REFLECT - Border will be mirror reflection of the border elements, like this : fedcba|abcdefgh|hgfedcb
    cv2.BORDER_REFLECT_101 or cv2.BORDER_DEFAULT - Same as above, but with a slight change, like this : gfedcb|abcdefgh|gfedcba
    cv2.BORDER_REPLICATE - Last element is replicated throughout, like this: aaaaaa|abcdefgh|hhhhhhh
    cv2.BORDER_WRAP - Can’t explain, it will look like this : cdefgh|abcdefgh|abcdefg
"""
def boder_control (bc):
    return {
        'constant': cv2.BORDER_CONSTANT,
        'reflect': cv2.BORDER_REFLECT,
        'replicate': cv2.BORDER_REPLICATE,
        'wrap': cv2.BORDER_WRAP,
    }[bc]

def convolution(src_img, mask, pad_type=VALID_CONSTANT, border_type='reflect'):
    assert len(src_img.shape) >= 2
    assert [dim > 0 for dim in src_img.shape]
    assert [dim > 0 for dim in mask.shape]

    src_type = src_img.dtype

    if len(src_img.shape) > 2:
        rows, cols, channels = src_img.shape
    elif len(src_img.shape) > 1:
        rows, cols = src_img.shape
        channels = 1
        src_img = cv2.merge((src_img, src_img))
    else:
        rows = src_img.shape



    if len(mask.shape) > 1:
        rows_mask, cols_mask = mask.shape
        type_border = boder_control(border_type)

        cols_pad = int(cols_mask/2)
        rows_pad = int(rows_mask/2)
        max_cols = cols
        max_rows = rows

        if pad_type == SAME_CONSTANT:
            dst_img = np.zeros(src_img.shape).astype(np.float64)
            src_img = cv2.copyMakeBorder(src_img, rows_pad, rows_pad, cols_pad, cols_pad, type_border, value=[0,0,0])
            max_cols = cols + cols_pad
            max_rows = rows + rows_pad
        elif pad_type == VALID_CONSTANT:
            if channels > 1:
                new_shape = (rows-2*rows_pad, cols-2*cols_pad, channels)
            elif channels == 1:
                new_shape = (rows-2*rows_pad, cols-2*cols_pad, channels+1)

            dst_img = np.zeros(new_shape).astype(np.float64)
            max_cols = cols - cols_pad
            max_rows = rows - rows_pad



    # Flipping mask
    mask = np.flipud(np.fliplr(mask))

    for c in range(channels):
        for x in range(cols_pad, max_cols):
            for y in range(rows_pad, max_rows):
                roi = src_img[y - rows_pad:y + rows_pad + 1, x - cols_pad:x + cols_pad + 1, c]

                sum = (roi * mask).sum()

                dst_img[y - rows_pad, x - cols_pad, c] = sum


    if channels == 1:
        dst_img, _ = cv2.split(dst_img)

    return dst_img.astype(src_type)
