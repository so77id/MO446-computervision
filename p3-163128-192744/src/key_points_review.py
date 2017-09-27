from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import cv2
import numpy as np

import sys
import argparse

import utils.img_utils as iu


DEBUG=False

def main(argv):
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input_list',  help='Input list',    required=True)
    parser.add_argument('-if', '--input_folder',  help='Input folder',    required=True)
    #parser.add_argument('-o', '--output_image',  help='Output image',    required=True)
    parser.add_argument('-d',  '--debug',    help='Debuggin mode', action='store_true')
    ARGS = parser.parse_args()

    global DEBUG
    DEBUG = ARGS.debug

    input_list = open(ARGS.input_list, 'r')

    frame0 = None
    frame1 = None

    for line in input_list:
        file_name = ARGS.input_folder + "/" + line.split(" ")[-1].split("\n")[0]
        print(file_name)
        img  = cv2.imread(file_name)
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

        if frame0 is None:
            frame0 = img

        sift = cv2.xfeatures2d.SIFT_create()
        key_points = sift.detect(gray)

        frame1 = cv2.drawKeypoints(gray,key_points, gray)

        grid = iu.create_grid(frame0, frame1)

        if DEBUG:
            cv2.imshow("grid", grid)
            # cv2.imshow("DIFF", diff)

            cv2.waitKey(0)
            cv2.destroyAllWindows()

        frame0 = frame1

if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))