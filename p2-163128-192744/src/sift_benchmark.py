from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import matplotlib.pyplot as plt
import sys
import time
import numpy as np
import cv2
import argparse


import utils.features2D as f2d


DEBUG=False

def create_graphic(graphic_out, x, y0, y1, llabel0, llabel1, ylabel, xlabel):
    plt.plot(x, y0, 'bx', label=llabel0)
    plt.plot(x, y1, 'rx', label=llabel1)

    plt.legend(loc='upper left', shadow=False)

    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    plt.grid(True)


    plt.savefig(graphic_out)
    plt.gcf().clear()

def get_time(f, *args):
    start = time.time()
    ret = f(*args)
    end = time.time()
    return (end-start)*1000.0, ret



def main(argv):
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-i',  '--input_list',    help='Input file', required=True)
    parser.add_argument('-g1',  '--graph_out_1',    help='file name for time benchmark graph', required=True)
    parser.add_argument('-g2',  '--graph_out_2',    help='file name for number of key points benchmar graph', required=True)
    parser.add_argument('-d',  '--debug',    help='Debuggin mode', action='store_true')
    args = parser.parse_args()

    global DEBUG
    DEBUG = args.debug

    input_list_files = open(args.input_list, 'r')

    img_sz = []
    tmp_own = []
    tmp_cv2 = []

    kps_own = []
    kps_cv2 = []


    for line in input_list_files:
        file_name, size =  line.split(" ")
        img = cv2.imread(file_name)
        print("Executing our sift")
        time_own, [kp0, desc0] = get_time(f2d.SIFT, img)

        print("Executing opencv sift")
        time_cv2, [kp1, desc1] = get_time(f2d.CVSIFT, img)

        split = size.split("x")
        size = int(split[0])*int(split[1])

        img_sz.append(size)
        # Times
        tmp_own.append(time_own)
        tmp_cv2.append(time_cv2)
        # Kps
        kps_own.append(kp0.shape[0])
        kps_cv2.append(kp1.shape[0])


        print('\t Imagen:',line.split('\n')[0])
        print ('\t\t Our SIFT:', time_own, "# kps:", kp0.shape[0])
        print ('\t\t OpenCV SIFT:', time_cv2, "# kps:", kp1.shape[0])

    create_graphic(args.graph_out_1, img_sz, tmp_own, tmp_cv2, "Our", "opencv", "Time (ms)", "Image sizes(pixels)")
    create_graphic(args.graph_out_2, img_sz, kps_own, kps_cv2, "Our", "opencv", "# Key points", "Image sizes(pixels)")



if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))