from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

ply_header = '''ply
format ascii 1.0
element vertex %(p_num)d
property float x
property float y
property float z
property uchar red
property uchar green
property uchar blue
end_header
'''

RANK = 3

def write_ply(filename, points, colors):
    points = points[:,:RANK]

    concat = np.concatenate((points,colors.transpose()), axis=1)

    with open(filename, 'wb') as f:
        print(concat.shape[0])
        f.write((ply_header % dict(p_num=concat.shape[0])).encode('utf-8'))
        np.savetxt(f, concat, fmt='%f %f %f %d %d %d ')