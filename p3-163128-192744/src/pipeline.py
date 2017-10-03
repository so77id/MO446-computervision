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
import utils.sfm as usfm


DEBUG=False


def main(argv):
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input_video',  help='Input video',    required=True)
    parser.add_argument('-d',  '--debug',    help='Debuggin mode', action='store_true')
    ARGS = parser.parse_args()

    global DEBUG
    DEBUG = ARGS.debug

    cap = cv2.VideoCapture(ARGS.input_video)
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            return

        gray0 = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        _, p0 = kps.detect_points(gray0)

        print(n_frames)
        print(p0.shape)

        U = np.zeros((n_frames, p0.shape[0]))
        V = np.zeros((n_frames, p0.shape[0]))
        S = np.ones((p0.shape[0],1))

        U[0,:] = p0[:,0,0]
        V[0,:] = p0[:,0,1]

        n_frame = 0
        while(1):
            ret, frame = cap.read()
            if not ret:
                break
            # print("Frame:", n_frame)
            n_frame+=1

            gray1 = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
            p1, st = klt.optical_flow(gray0, gray1, p0, mode="cv2")

            U[n_frame, :] = p1[:, 0, 0]
            V[n_frame, :] = p1[:, 0, 1]
            S*=st

        S = S.flatten()
        U = U[:, S==1]
        V = V[:, S==1]

        W = np.concatenate((U,V), axis=0)

        M, S = usfm.sfm(W)

        M_0 = np.matrix(np.zeros((3,3)))
        M_0[:,0] = M[0,:].transpose()
        M_0[:,1] = M[n_frames,:].transpose()
        M_0[:,2] = np.cross(M[0,:], M[n_frames,:]).transpose()

        print(M_0)
        print(M_0.shape)

        # print(M)
        # print(S)



    cap.release()

if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))