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
property uchar blue
property uchar green
property uchar red
end_header
'''



def write_ply(filename, points, colors, RANK = 4):
    if RANK == 4:
        points = points[:,:(RANK-1)]

    concat = np.concatenate((points,colors.transpose()), axis=1)

    with open(filename, 'wb') as f:
        f.write((ply_header % dict(p_num=concat.shape[0])).encode('utf-8'))
        np.savetxt(f, concat, fmt='%f %f %f %d %d %d ')
