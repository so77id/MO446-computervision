from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import cv2
import numpy as np
from numpy.linalg import lstsq
from numpy.linalg import cholesky
from numpy.linalg import inv


def g_t(a, b):
    return np.array([a[0,0]*b[0,0],
                    a[0,0]*b[0,1] + a[0,1]*b[0,0],
                    a[0,0]*b[0,2]+a[0,2]*b[0,0],
                    a[0,1]*b[0,1],
                    a[0,1]*b[0,2]+a[0,2]*b[0,1],
                    a[0,2]*b[0,2]])

def get_c(F):
    o = np.ones((2*F))
    z = np.zeros((F))
    return np.concatenate((o,z))

def get_g(M):
    F = M.shape[0]//2

    G = np.zeros((3*F, 6))

    for i in range(F):
        G[i, :] = g_t(M[i], M[i])
        G[i+F, :] = g_t(M[i+F], M[i+F])
        G[i+(2*F), :] = g_t(M[i], M[i+F])

    return G

def sfm(W):

    a_f = np.mean(W, axis=1).reshape(W.shape[0], 1)
    W_aprox = np.matrix(W - a_f)

    U, s, V = np.linalg.svd(W_aprox)

    U_ = U[:,:3]
    s_ = np.matrix(np.diag(s[:3]))
    V_ = V[:3,:]

    print(U_.shape)
    print(s_.shape)
    print(V_.shape)

    M_hat = U_
    S_hat = s_ * V_

    # Number of frames
    F = M_hat.shape[0]//2

    G = get_g(M_hat)
    c = get_c(F)

    l = lstsq(G, c)[0]
    print(l.shape)
    L = np.matrix([[l[0], l[1], l[2]],
                   [l[1], l[3], l[4]],
                   [l[2], l[4], l[5]]])

    A = cholesky(L)

    M = M_hat * A
    S = inv(A) * S_hat

    return M, S