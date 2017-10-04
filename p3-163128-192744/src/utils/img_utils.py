from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import cv2
import numpy as np

def get_write_instance(video_out, cap_in, shape=None):
    fps = cap_in.get(cv2.CAP_PROP_FPS)
    frames = int(cap_in.get(cv2.CAP_PROP_FRAME_COUNT))
    codec = cv2.VideoWriter_fourcc(*'MP4V')
    if shape == None:
        width = cap_in.get(cv2.CAP_PROP_FRAME_WIDTH)   # float
        height = cap_in.get(cv2.CAP_PROP_FRAME_HEIGHT) # float
    else:
        width = shape[0]   # float
        height = shape[1] # float


    return cv2.VideoWriter(video_out, int(codec), fps, (int(width),int(height)))

def create_grid(img0, img1):
    assert(img0.shape == img1.shape)

    return np.concatenate((img0, img1), axis=1)

def draw_kp_and_write(out_file, kps, img):
    img = cv2.drawKeypoints(img, kps, img)
    cv2.imwrite(out_file, img)
    return