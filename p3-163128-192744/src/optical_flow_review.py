from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import cv2
import numpy as np

import sys
import argparse

import utils.img_utils as iu
import utils.key_points as kps
import utils.klt as klt

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

    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)   # float
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT) # float

    video_out = iu.get_write_instance(ARGS.output_video, cap, (2*width, height))

    if cap.isOpened():
        ret, frame0 = cap.read()
        frame0_ = frame0
        if not ret:
            return

        gray0 = cv2.cvtColor(frame0,cv2.COLOR_BGR2GRAY)
        gray0_ = gray0
        _, p0 = kps.detect_points(gray0)
        p0_ = p0

        # Create some random colors
        color = np.random.randint(0,255,(p0.shape[0],3))
        print(p0.shape)
        # p0 = cv2.goodFeaturesToTrack(gray0, mask = None, **feature_params)
        print(p0.shape)

        # Create a mask image for drawing purposes
        mask = np.zeros_like(frame0)
        mask_ = np.zeros_like(frame0_)

        img = None
        img_ = None
        count = 0
        while(1):
                count+=1
                print(count)
                ret, frame1 = cap.read()
                frame1_ = frame1
                if not ret:
                    break

                gray1 = cv2.cvtColor(frame1,cv2.COLOR_BGR2GRAY)
                gray1_ = gray1

                p1, st = klt.optical_flow(gray0, gray1, p0, mode="own")
                p1_, st_ = klt.optical_flow(gray0_, gray1_, p0_, mode="cv2")

                good_new = p1[st==1]
                good_old = p0[st==1]

                good_new_ = p1_[st_==1]
                good_old_ = p0_[st_==1]

                for i,(new,old) in enumerate(zip(good_new,good_old)):
                    a,b = new.ravel()
                    c,d = old.ravel()
                    mask = cv2.line(mask, (a,b),(c,d), color[i].tolist(), 2)
                    frame1 = cv2.circle(frame1,(a,b),5,color[i].tolist(),-1)
                img = cv2.add(frame1,mask)

                for i,(new,old) in enumerate(zip(good_new_,good_old_)):
                    a,b = new.ravel()
                    c,d = old.ravel()
                    mask_ = cv2.line(mask_, (a,b),(c,d), color[i].tolist(), 2)
                    frame1_ = cv2.circle(frame1_,(a,b),5,color[i].tolist(),-1)
                img_ = cv2.add(frame1_,mask_)

                grid = iu.create_grid(img, img_)


                video_out.write(grid)

                frame0 = frame1
                gray0 = gray1
                p0 = p1

                frame0_ = frame1_
                gray0_ = gray1_
                p0_ = p1_

        if DEBUG:
            cv2.imshow("grid", grid)

            cv2.waitKey(0)
            cv2.destroyAllWindows()


    cap.release()
    video_out.release()

if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))