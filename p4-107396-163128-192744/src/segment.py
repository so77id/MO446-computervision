from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import sys
import cv2
import argparse
import numpy as np
import utils.io as io
from glob import glob
from os import path, makedirs
from utils.regions import segment_image, connected_components

def colorize(img, img_labeled, k):
    new_img = np.zeros_like(img)

    for i in range(k):
        color = img[img_labeled == i].mean(axis=0)

        new_img[img_labeled == i] = color

    return new_img

def main(argv):

    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-i',  '--input_folder',     help='Input folder',                          required=True)
    parser.add_argument('-k',  '--n_clusters',       help='Number of cluster',                     required=True,)
    parser.add_argument('-b',  '--blur_window_size', help='Blurring before segmenting',            required=False)
    parser.add_argument('-sf', '--segments_file',    help='Output file of the segments',         required=False)
    parser.add_argument('-id', '--images_dir',       help='Output folder of the segmented images', required=False)
    parser.add_argument('-n',  '--normalize',        help='Normalizing before segmenting',         action='store_true')
    parser.add_argument('-d',  '--debug',            help='Debuggin mode',                         action='store_true')
    ARGS = parser.parse_args()


    DEBUG = ARGS.debug
    k = int(ARGS.n_clusters)
    b = int(ARGS.blur_window_size) if ARGS.blur_window_size is not None else 0

    if ARGS.images_dir is not None:
        out_img_dir = ARGS.images_dir
        if not path.isdir(out_img_dir):
            makedirs(out_img_dir)
    else:
        out_img_dir = None

    img_regions = {}
    for img_file in glob(path.join(ARGS.input_folder, '*.jpg')):

        ori = cv2.imread(img_file)

        regions = segment_image(ori, k, b, ARGS.normalize)
        cc, k_cc = connected_components(regions, k)

        img_regions[path.basename(img_file).split('.')[0]] = (cc, k_cc)

        print('Processed %s with %s regions.' % (path.basename(img_file), str(k_cc)))

        if out_img_dir is not None:
                        
            cc_img = colorize(ori, cc, k_cc).astype(np.uint8)
            out_file = path.join(out_img_dir, path.basename(img_file))
            cv2.imwrite(out_file, cc_img)

    if(ARGS.segments_file is not None):
        io.write(img_regions, output_file=ARGS.segments_file)


    if DEBUG:
        cv2.imshow("ori", cc_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))