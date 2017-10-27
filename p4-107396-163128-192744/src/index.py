from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import os
import cv2
import sys
import argparse

import utils.io as io
import utils.bovw as bovw
import utils.descriptor as desc

DEBUG=False

def main(argv):
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-il', '--input_list',  help='Input list',    required=True)
    parser.add_argument('-if', '--input_folder',  help='Input folder',    required=True)
    parser.add_argument('-de', '--descriptor_name',  help='Descriptor name',    required=True)
    parser.add_argument('-rf', '--regions_file',  help='Regions input file',    required=True)
    parser.add_argument('-nf', '--n_features',  help='Number of features',    required=True)
    parser.add_argument('-ro', '--region_descriptors_output_file',  help='Output file',    required=True)
    parser.add_argument('-do', '--descriptors_output_file',  help='Output file',    required=True)
    parser.add_argument('-vwo', '--visual_words_output_file',  help='Output file',    required=True)
    parser.add_argument('-d',  '--debug',    help='Debuggin mode', action='store_true')
    ARGS = parser.parse_args()

    global DEBUG
    DEBUG = ARGS.debug

    regions = io.read(ARGS.regions_file)

    region_descriptors_output_file = ARGS.region_descriptors_output_file
    descriptors_output_file = ARGS.descriptors_output_file
    visual_words_output_file = ARGS.visual_words_output_file

    if os.path.isfile(region_descriptors_output_file) is False:
        with open(ARGS.input_list, 'r') as input_list_files:
            descriptors = {}
            for line in input_list_files:
                print("Reading:", line)
                file_name = "{}/{}".format(ARGS.input_folder, line).split("\n")[0]
                img = cv2.imread(file_name)

                labeled_image, n_regions = regions[line.split(".")[0]]

                img_descriptor = desc.get_descriptors(img, labeled_image, n_regions, descriptor_name=ARGS.descriptor_name)
                descriptors[line.split(".")[0]] = img_descriptor

            io.write(descriptors, output_file=region_descriptors_output_file)
    else:
        descriptors = io.read(region_descriptors_output_file)

    img_descriptors, visual_words = bovw.describe_images(descriptors, int(ARGS.n_features))

    io.write(img_descriptors, output_file=descriptors_output_file)
    io.write(visual_words, output_file=visual_words_output_file)

if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
