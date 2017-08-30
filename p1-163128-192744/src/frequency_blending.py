from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import numpy as np
import cv2
import argparse

import utils.conv as conv
import utils.blending as blending
import utils.dft as dft
import utils.pyramid as pyramid

DEBUG=False




def main(argv):
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-i1', '--input_file1',  help='Input file',    required=True)
    parser.add_argument('-i2', '--input_file2',  help='Input file',    required=True)
    parser.add_argument('-m',  '--mask',         help='Input file',    required=True)
    parser.add_argument('-o',  '--output_file',  help='Output file',   required=True)
    parser.add_argument('-ps', '--pyramid_size', help='Pyramid size',    required=False)
    parser.add_argument('-a', '--approach', help='Method used (1 or 2)',    required=True)
    parser.add_argument('-d',  '--debug',        help='Debuggin mode', action='store_true')
    ARGS = parser.parse_args()

    global DEBUG
    DEBUG = ARGS.debug

    img1 = cv2.imread(ARGS.input_file1, 0)
    img2 = cv2.imread(ARGS.input_file2, 0)

    mask = cv2.imread(ARGS.mask, 0)
    mask = np.uint8(mask/mask.max())

    if(ARGS.approach == 1):
        assert(pyramid_size is not None)
        pyramid_size = int(ARGS.pyramid_size)
        kernel = conv.gaussian_kernel(13)
        blend_img = blending.freq_pyamid_blending(img1, img2, mask, pyramid_size, kernel)
    else:
        blend_img = blending.freq_concat_blending(img1, img2, mask)

    cv2.imwrite(ARGS.output_file, blend_img)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))