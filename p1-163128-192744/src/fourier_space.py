from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import argparse

import numpy as np
import cv2
import utils.dft

DEBUG=True

def main(argv):
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-i',  '--input_file',    help='Input file', required=True)
    parser.add_argument('-o',  '--output_file',    help='Output file', required=True)
    parser.add_argument('-t',  '--threshold',    help='Threshold')
    parser.add_argument('-r',  '--decreasing',    help='Decreasing order?', action='store_true')
    parser.add_argument('-p',  '--phase',    help='Filter phase?', action='store_true')
    parser.add_argument('-d',  '--debug',    help='Debuggin mode', action='store_true')
    args = parser.parse_args()

    global DEBUG
    DEBUG = args.debug

    threshold = args.threshold if args.threshold is None else float(args.threshold)

    input_img = np.float32(cv2.imread(args.input_file, 0))
    dft_m, dft_p = utils.dft.dft_mp(input_img)


    dft_m_f, dft_p_f = utils.dft.filter(dft_m, dft_p, threshold, args.decreasing, args.phase)
    cv2.imwrite(args.output_file, utils.dft.idft_mp(dft_m_f, dft_p_f))

if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
