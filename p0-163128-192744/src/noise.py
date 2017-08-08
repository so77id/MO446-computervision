from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import cv2
import numpy as np
import argparse

import color_planes as cp

DEBUG=False
MEAN=0
STD=30

def a_function(original_img):
    blue, green, red = cv2.split(original_img)

    noise = np.empty(green.shape, dtype=np.uint8)
    noise = cv2.randn(noise, MEAN, STD)
    green_noise = (green + noise).astype(np.uint8)

    noise_img = cv2.merge((blue, green_noise, red))

    return noise_img


def b_function(original_img):
    blue, green, red = cv2.split(original_img)

    noise = np.empty(blue.shape, dtype=np.uint8)
    noise = cv2.randn(noise, MEAN, STD)
    blue_noise = (blue + noise).astype(np.uint8)

    noise_img = cv2.merge((blue_noise, green, red))

    return noise_img



def main(argv):
  # Parse arguments
  parser = argparse.ArgumentParser()
  parser.add_argument('-i',  '--input_file',    help='Input file', required=True)
  parser.add_argument('-oa', '--output_file_a', help='Output file from a question', required=True)
  parser.add_argument('-ob', '--output_file_b', help='Output file from b question', required=True)
  args = parser.parse_args()

  input_img = cv2.imread(args.input_file)

  a_img = a_function(input_img)
  cv2.imwrite(args.output_file_a, a_img)

  b_img = b_function(input_img)
  cv2.imwrite(args.output_file_b, b_img)


  if DEBUG:
    cv2.imshow("Input image", input_img)
    cv2.imshow("A function", a_img)
    cv2.imshow("B function", b_img)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
  sys.exit(main(sys.argv[1:]))