from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import cv2
import numpy as np

import sys
import argparse

import utils.img_utils as iu
import utils.key_points as kps

DEBUG=False

feature_params = dict(  maxCorners = 100,
                        qualityLevel = 0.3,
                        minDistance = 7,
                        blockSize = 7 )

# Parameters for lucas kanade optical flow
lk_params = dict( winSize  = (15,15),
                  maxLevel = 2,
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))



def main(argv):
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input_video',  help='Input video',    required=True)
    parser.add_argument('-o', '--output_video',  help='Output video',    required=True)
    parser.add_argument('-d',  '--debug',    help='Debuggin mode', action='store_true')
    ARGS = parser.parse_args()

    global DEBUG
    DEBUG = ARGS.debug


    cap = cv2.VideoCapture(ARGS.input_video)
    video_out = iu.get_write_instance(ARGS.output_video, cap)

    if cap.isOpened():
        ret, frame0 = cap.read()
        if not ret:
            return

        gray0 = cv2.cvtColor(frame0,cv2.COLOR_BGR2GRAY)

        _, p0 = kps.detect_points(gray0)

        # Create some random colors
        color = np.random.randint(0,255,(p0.shape[0],3))
        print(p0.shape)
        # p0 = cv2.goodFeaturesToTrack(gray0, mask = None, **feature_params)
        print(p0.shape)

        # Create a mask image for drawing purposes
        mask = np.zeros_like(frame0)
        img = None
        count = 0
        while(1):
                count+=1
                print(count)
                ret, frame1 = cap.read()
                if not ret:
                    break

                gray1 = cv2.cvtColor(frame1,cv2.COLOR_BGR2GRAY)

                p1, st, err = cv2.calcOpticalFlowPyrLK(gray0, gray1, p0, None, **lk_params)

                good_new = p1[st==1]
                good_old = p0[st==1]

                # draw the tracks
                for i,(new,old) in enumerate(zip(good_new,good_old)):
                    a,b = new.ravel()
                    c,d = old.ravel()
                    mask = cv2.line(mask, (a,b),(c,d), color[i].tolist(), 2)
                    frame1 = cv2.circle(frame1,(a,b),5,color[i].tolist(),-1)
                img = cv2.add(frame1,mask)

                video_out.write(img)

                frame0 = frame1
                gray0 = gray1
                p0 = p1


        if DEBUG:
            cv2.imshow("grid", img)

            cv2.waitKey(0)
            cv2.destroyAllWindows()

    cap.release()
    video_out.release()

if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))