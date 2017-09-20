from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import numpy as np
import cv2
import argparse
from time import time
from utils.transform import transform
import utils.features2D as f2d
from utils.ransac import ransac
import utils.video_utils as vutils
import utils.descriptor_matcher as d_match
from utils.model import AffineTransformationModel
from utils.model import ProjectiveTransformationModel

DEBUG=False

def main(argv):
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input_video',  help='Input video',    required=True)
    parser.add_argument('-o', '--output_video',  help='Output video',    required=True)
    parser.add_argument('-oc', '--output_comparer_video',  help='Output comparer video',    required=True)
    parser.add_argument('-kp_t', '--kp_threshold',  help='Theshold used in key point selection',    required=True)
    parser.add_argument('-d',  '--debug',        help='Debuggin mode', action='store_true')
    ARGS = parser.parse_args()

    global DEBUG
    DEBUG = ARGS.debug

    cap = cv2.VideoCapture(ARGS.input_video)
    video_out = vutils.get_write_instance(ARGS.output_video, cap)

    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)   # float
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT) # float
    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    video_comp_out = vutils.get_write_instance(ARGS.output_comparer_video, cap, (2*width, height))

    if cap.isOpened():
        ret, frame = cap.read()
        kp0, des0 = f2d.SIFT(frame)

        p = np.indices(frame.shape[:2]).swapaxes(0,2).reshape((np.prod(frame.shape[:2]),2), order = 'F')

        i = 2
        
        t_acc = 0
        while(1):
            start = time()
            ret, frame = cap.read()

            if not ret:
                break

            kp1, des1 = f2d.SIFT(frame)
            matches = d_match.kp_matcher(des0, des1, float(ARGS.kp_threshold))
            print("kps:",kp1.shape)

            y = kp0[matches[matches < len(kp0)]]
            x = kp1[matches < len(kp0)]
            print("matches:",x.shape)

            model, _ = ransac(x, y, AffineTransformationModel(), 100, 1.0, int(x.shape[0] * 0.8))
            img = transform(frame, model, p)

            img = img.astype(np.uint8)
            grid_image = vutils.create_grid(frame, img).astype(np.uint8)

            if DEBUG:
                cv2.imwrite('./output/corrected/frame%d.png' % (i), img)
                cv2.imwrite('./output/original/_frame%d.png' % (i), frame)
                cv2.imwrite('./output/composed/grid%d.png' % (i), grid_image)

            video_out.write(img)
            video_comp_out.write(grid_image)

            kp0 = model.predict(kp1)
            des0 = des1
            end = time()
            t_acc += end - start
            print("Frame %04d/%04d\tProc. time: %.2fs\tTotal time: %dm%.2ds" % (i, length, end - start, t_acc // 60, round(t_acc % 60)))
            i += 1

        

        cap.release()
        video_out.release()
        video_comp_out.release()


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))