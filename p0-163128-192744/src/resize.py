from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import cv2
import numpy as np
import argparse

CROP_SIZE=512
DEBUG=False

def resize(img):
    width, height, channels = img.shape

    if width > CROP_SIZE or height > CROP_SIZE:
        if(width > height):
            scale = float(CROP_SIZE)/float(width)
        else:
            scale = float(CROP_SIZE)/float(height)


        img = cv2.resize(img, None, fx=scale, fy=scale)
    return img



def main(argv):
  # Parse arguments
  parser = argparse.ArgumentParser()
  parser.add_argument('-i',  '--input_file',    help='Input file', required=True)
  parser.add_argument('-o',  '--output_file', help='Output file', required=True)
  args = parser.parse_args()

  input_img = cv2.imread(args.input_file)

  output_img = resize(input_img)
  cv2.imwrite(args.output_file, output_img)


  if DEBUG:
    cv2.imshow("Input image", input_img)
    cv2.imshow("Output image", output_img)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
  sys.exit(main(sys.argv[1:]))