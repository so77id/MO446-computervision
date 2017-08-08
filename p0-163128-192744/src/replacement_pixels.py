from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import cv2
import numpy as np
import argparse

import color_planes as cp

DEBUG=False
MID_RANGE=50


def a_function(original_img):
    A = cp.b_function(original_img) # Green Channel
    B = cp.c_function(original_img) # Red Channel

    height, width, channels = original_img.shape

    height_bottom_range = int(height/2) - MID_RANGE
    height_upper_range = int(height/2) + MID_RANGE

    width_bottom_range = int(width/2) - MID_RANGE
    width_upper_range = int(width/2) + MID_RANGE

    B[height_bottom_range:height_upper_range, width_bottom_range:width_upper_range] = A[height_bottom_range:height_upper_range, width_bottom_range:width_upper_range]

    return B




def b_function(original_img, channel_modified):
    blue, green, _ = cv2.split(original_img)
    original_img = cv2.merge((blue, green, channel_modified))
    return original_img


def main(argv):
  # Parse arguments
  parser = argparse.ArgumentParser()
  parser.add_argument('-i',  '--input_file',    help='Input file', required=True)
  parser.add_argument('-oa', '--output_file_a', help='Output file from a question', required=True)
  parser.add_argument('-ob', '--output_file_b', help='Output file from b question', required=True)
  args = parser.parse_args()

  input_img = cv2.imread(args.input_file)

  a_img = a_function(input_img)
  cv2.imwrite(args.output_file_a ,a_img)

  b_img = b_function(input_img, a_img)
  cv2.imwrite(args.output_file_b ,b_img)

  if DEBUG:
    cv2.imshow("INPUT", input_img)
    cv2.imshow("A FUNCTION", a_img)
    cv2.imshow("B FUNCTION", b_img)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
  sys.exit(main(sys.argv[1:]))