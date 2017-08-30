from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import matplotlib.pyplot as plt
import sys
import time
import numpy as np
import cv2
import argparse
import utils.conv as conv


DEBUG=False

def create_graphic(graphic_out, x, y0, y1, llabel0, llabel1, ylabel, xlabel):
    plt.plot(x, y0, 'bx', label=llabel0)
    plt.plot(x, y1, 'rx', label=llabel1)

    plt.legend(loc='upper left', shadow=False)

    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    plt.grid(True)

    if(DEBUG):
        plt.show()

    plt.savefig(graphic_out)
    plt.gcf().clear()

def get_time(f, *args):
    start = time.time()
    ret = f(*args)
    end = time.time()
    return (end-start)*1000.0, ret

def create_kernel_list(n_kernels):
    kernels = []
    size = 3
    for i in range(n_kernels):
        kernels.append(conv.gaussian_kernel(size))
        size += 2

    return kernels


def main(argv):
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-i',  '--input_list',    help='Input file', required=True)
    parser.add_argument('-nk',  '--n_kernels',    help='Number of kernels', required=True)
    parser.add_argument('-g1',  '--graph_out_1',    help='file name for image size benchmark graph', required=True)
    parser.add_argument('-g2',  '--graph_out_2',    help='file name for kernel size benchmark graph', required=True)
    parser.add_argument('-d',  '--debug',    help='Debuggin mode', action='store_true')
    args = parser.parse_args()

    global DEBUG
    DEBUG = args.debug

    # Get list of kernels for kernel size benchmark
    kernels = create_kernel_list(int(args.n_kernels))
    # Kernel used for image size benchmark
    kernel = kernels[0]

    input_list_files = open(args.input_list, 'r')

    # Vars for image size benchmark
    img_sz = []
    img_sz_mconv_time = []
    img_sz_cv2conv_time = []

    # Vars for kernel size benchmark
    kernel_sz = []
    kernel_sz_mconv_time = []
    kernel_sz_cv2conv_time = []


    first_iter = True

    # Image size benchmark
    print("\tImage size benchmark")
    print("\t--------------------")
    for line in input_list_files:
        file_name, size =  line.split(" ")
        img = cv2.imread(file_name, 0)

        time_own, output1_img = get_time(conv.convolution, img, kernel)
        # time_own, output1_img = get_time(cv2.filter2D, img,-1,kernel)
        time_cv2, output2_img = get_time(cv2.filter2D, img,-1,kernel)

        split = size.split("x")
        size = int(split[0])*int(split[1])

        # if is the first image save this for kernel benchmark
        if first_iter == True:
            first_image = img
            first_size = size
            first_iter = False

        img_sz.append(size)
        img_sz_mconv_time.append(time_own)
        img_sz_cv2conv_time.append(time_cv2)

        print('\t Imagen:',line.split('\n')[0])
        print ('\t\t Our Convolution:', time_own)
        print ('\t\t OpenCV Convolution:', time_cv2)
        # diff = output1_img - output2_img

        if DEBUG:
            cv2.imshow("INPUT", img)
            cv2.imshow("OUTPUT1", output1_img)
            cv2.imshow("OUTPUT2", output2_img)
            # cv2.imshow("DIFF", diff)

            cv2.waitKey(0)
            cv2.destroyAllWindows()

    create_graphic(args.graph_out_1, img_sz, img_sz_mconv_time, img_sz_cv2conv_time, "Our", "opencv", "Time (ms)", "Image sizes(pixels)")


    # Kernel size benchmark
    print("\tKernel size benchmark")
    print("\t---------------------")
    for kernel in kernels:
        time_own, output1_img = get_time(conv.convolution, first_image, kernel)
        # time_own, output1_img = get_time(cv2.filter2D, first_image,-1,kernel)
        time_cv2, output2_img = get_time(cv2.filter2D, first_image,-1,kernel)

        size, _ = kernel.shape

        kernel_sz.append(size)
        kernel_sz_mconv_time.append(time_own)
        kernel_sz_cv2conv_time.append(time_cv2)

        print('\t Kernel size:', size)
        print ('\t\t Our Convolution:', time_own)
        print ('\t\t OpenCV Convolution:', time_cv2)

        if DEBUG:
            cv2.imshow("INPUT", first_image)
            cv2.imshow("OUTPUT1", output1_img)
            cv2.imshow("OUTPUT2", output2_img)

    create_graphic(args.graph_out_2, kernel_sz, kernel_sz_mconv_time, kernel_sz_cv2conv_time, "Our", "opencv", "Time (ms)", "Kernel sizes(pixels)")

if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))