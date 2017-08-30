from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import matplotlib.pyplot as plt
import sys
import time
import numpy as np
import cv2
import argparse
import utils.pyramid as pyramid
import utils.conv as conv


DEBUG=False


def main(argv):
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-i',  '--input_img', help='Input file', required=True)
    parser.add_argument('-ps',  '--pyramid_size', help='Pyramid size', required=True)
    parser.add_argument('-opl',  '--output_pl', help='Pyramid laplacian file', required=True)
    parser.add_argument('-opg',  '--output_pg', help='Pyramid laplacian file', required=True)
    parser.add_argument('-d',  '--debug',    help='Debuggin mode', action='store_true')
    ARGS = parser.parse_args()

    global DEBUG
    DEBUG = ARGS.debug
    src_img = cv2.imread(ARGS.input_img, 0)

    kernel = conv.gaussian_kernel(15)
    #kernel = np.array([[1,1,1],[1,1,1],[1,1,1]], dtype=np.float64) * 1/9
    py_gaussian = pyramid.GaussianPyramid(src_img, int(ARGS.pyramid_size), kernel)
    py_laplacian = pyramid.LaplacianPyramid(py_gaussian)

    pg_img = py_gaussian.composition()
    pl_img = py_laplacian.composition()

    cv2.imwrite(ARGS.output_pg, pg_img)
    cv2.imwrite(ARGS.output_pl, pl_img)
    if DEBUG:
            cv2.imshow("pg", pg_img)
            cv2.imshow("pl", pl_img)

            cv2.waitKey(0)
            cv2.destroyAllWindows()

    # print("Gaussian Pyramid")
    # print("Up:")
    # for i in py_gaussian.range():
    #     img = py_gaussian.access(i)
    #     print("\tPyramid index:", i, "shape:", img.shape)

    #     if DEBUG:
    #         cv2.imshow("Pyramid index:{}".format(i), img)

    #         cv2.waitKey(0)
    #         cv2.destroyAllWindows()

    # print("Down:")

    # for i in py_gaussian.inv_range():
    #     img = py_gaussian.access(i)

    #     print("\tPyramid index:", i, "shape:", img.shape)

    #     if DEBUG:
    #         cv2.imshow("Pyramid index:{}".format(i), img)

    #         cv2.waitKey(0)
    #         cv2.destroyAllWindows()

    # print("Laplacian Pyramid")
    # print("Up:")
    # for i in py_laplacian.range():
    #     img = py_laplacian.access(i)
    #     print("\tPyramid index:", i, "shape:", img.shape)

    #     if DEBUG:
    #         cv2.imshow("Pyramid index:{}".format(i), img)

    #         cv2.waitKey(0)
    #         cv2.destroyAllWindows()

    # print("Down:")
    # for i in py_laplacian.inv_range():
    #     img = py_laplacian.access(i)
    #     print("\tPyramid index:", i, "shape:", img.shape)

    #     if DEBUG:
    #         cv2.imshow("Pyramid index:{}".format(i+1), img)

    #         cv2.waitKey(0)
    #         cv2.destroyAllWindows()




if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))