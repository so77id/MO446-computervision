from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import cv2
import numpy as np
import argparse

import color_planes as cp

DEBUG=False
MULTIPLEXER=10
N_SHIFT=2

def max(array):
    array = np.reshape(array, (-1))
    max = -float('inf')

    for bin in array:
        if bin > max:
            max = bin

    return max

def min(array):
    array = np.reshape(array, (-1))
    min = float('inf')

    for bin in array:
        if bin < min:
            min = bin

    return min


def mean(array):
    array = np.reshape(array, (-1))
    mean = array.sum()/array.shape[0]

    return mean

def std(array):
    array = np.reshape(array, (-1))
    std = np.sqrt((np.power(array - mean(array), 2)).sum()/(array.shape[0]-1))

    return std

def normalize(img):
    mean_ = mean(img)
    std_ = std(img)

    norm = (((img.astype(float) - mean_)/std_)*MULTIPLEXER) + mean_

    return norm.astype(np.uint8)


def shift_image(img):
    height, width = img.shape

    shift_matrix = np.float32([[1, 0, -N_SHIFT],[0, 1, 0]])
    shifted_img  = cv2.warpAffine(img, shift_matrix, (width, height))

    return shifted_img

def a_function(green_img):
    green_max = max(green_img)
    green_min = min(green_img)
    green_mean = mean(green_img)
    green_std = std(green_img)

    print("\t\tMax value:", green_max)
    print("\t\tMin value:", green_min)
    print("\t\tMean value:", green_mean)
    print("\t\tStd value:", green_std)

    return green_img

def b_function(green_img):
    return normalize(green_img)


def c_function(green_img):
    shifted_img = shift_image(green_img)
    substracted_img = green_img - shifted_img

    substracted_img[substracted_img > 255] = 255
    substracted_img[substracted_img < 0]   = 0

    return shifted_img, substracted_img


def main(argv):
  # Parse arguments
  parser = argparse.ArgumentParser()
  parser.add_argument('-i',  '--input_file',    help='Input file', required=True)
  parser.add_argument('-ob', '--output_file_b', help='Output file from b question', required=True)
  parser.add_argument('-oc0', '--output_file_c0', help='Output file from c 0 question', required=True)
  parser.add_argument('-oc1', '--output_file_c1', help='Output file from c 1 question', required=True)
  args = parser.parse_args()

  input_img = cv2.imread(args.input_file)
  green_img = cp.b_function(input_img) # Green image

  print("\tMetrics from original image:")
  a_function(green_img)

  b_img = b_function(green_img)
  cv2.imwrite(args.output_file_b, b_img)

  print("\tMetrics from normalized image:")
  a_function(b_img)

  c0_img, c1_img = c_function(green_img)
  cv2.imwrite(args.output_file_c0, c0_img)
  cv2.imwrite(args.output_file_c1, c1_img)

  print("\tMetrics from substracted image:")
  a_function(c1_img)


  if DEBUG:
    cv2.imshow("Input image", input_img)
    cv2.imshow("Input green", green_img)
    cv2.imshow("B function", b_img)
    cv2.imshow("C0 FUNCTION", c0_img)
    cv2.imshow("C1 FUNCTION", c1_img)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
  sys.exit(main(sys.argv[1:]))