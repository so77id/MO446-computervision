from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import cv2
import numpy as np

import sys
import argparse

import utils.img_utils as iu
import utils.key_points as kps


DEBUG=False

def main(argv):
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input_video',  help='Input video',    required=True)
    parser.add_argument('-d',  '--debug',    help='Debuggin mode', action='store_true')
    ARGS = parser.parse_args()

    global DEBUG
    DEBUG = ARGS.debug

    cap = cv2.VideoCapture(ARGS.input_video)

    if cap.isOpened():
        ret, frame0 = cap.read()
        if not ret:
            return

        gray0 = cv2.cvtColor(frame0,cv2.COLOR_BGR2GRAY)

        p0, _ = kps.detect_points(gray0)

        while(1):
                ret, frame1 = cap.read()
                if not ret:
                    break

                gray1 = cv2.cvtColor(frame1,cv2.COLOR_BGR2GRAY)

                p1, _ = kps.detect_points(gray1)

                frame1 = cv2.drawKeypoints(frame1, p1, frame1)

                grid = iu.create_grid(frame0, frame1)


                if DEBUG:
                    cv2.imshow("grid", grid)
                    # cv2.imshow("DIFF", diff)

                    cv2.waitKey(0)
                    cv2.destroyAllWindows()

                frame0 = frame1

if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))