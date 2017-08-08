from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import cv2
import numpy as np
import argparse

DEBUG=False

# Channel order in cv2 is BGR
def a_function(original_img):
  b,g,r = cv2.split(original_img)
  original_img = cv2.merge((r,g,b))
  # red = original_img[:,:,2]
  # blue = original_img[:,:,0]
  # original_img[:,:,0] = red
  # original_img[:,:,2] = blue

  return original_img


def b_function(original_img):
  _, g, _ = cv2.split(original_img)

  return g

def c_function(original_img):
  _, _, r = cv2.split(original_img)

  return r



def main(argv):
  # Parse arguments
  parser = argparse.ArgumentParser()
  parser.add_argument('-i',  '--input_file',    help='Input file', required=True)
  parser.add_argument('-oa', '--output_file_a', help='Output file from a question', required=True)
  parser.add_argument('-ob', '--output_file_b', help='Output file from b question', required=True)
  parser.add_argument('-oc', '--output_file_c', help='Output file from c question', required=True)
  args = parser.parse_args()

  input_img = cv2.imread(args.input_file)

  a_img = a_function(input_img)
  cv2.imwrite(args.output_file_a ,a_img)

  b_img = b_function(input_img)
  cv2.imwrite(args.output_file_b ,b_img)

  c_img = c_function(input_img)
  cv2.imwrite(args.output_file_c ,c_img)

  if DEBUG:
    cv2.imshow("INPUT", input_img)
    cv2.imshow("A FUNCTION", a_img)
    cv2.imshow("B FUNCTION", b_img)
    cv2.imshow("C FUNCTION", c_img)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
  sys.exit(main(sys.argv[1:]))