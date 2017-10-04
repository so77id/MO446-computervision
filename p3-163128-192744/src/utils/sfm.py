from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import cv2
import numpy as np
from numpy.linalg import lstsq
from numpy.linalg import cholesky
from numpy.linalg import inv
from scipy.linalg import sqrtm

RANK=3

# n (n+1) / 2
def comb(n):
    return n*(n+1)//2

def squareform_diagfill(arr1D):
    n = int(np.sqrt(arr1D.size*2))
    if (n*(n+1))//2!=arr1D.size:
        print("Size of 1D array not suitable for creating a symmetric 2D array!")
        return None
    else:
        R,C = np.triu_indices(n)
        out = np.zeros((n,n),dtype=arr1D.dtype)
        out[R,C] = arr1D
        out[C,R] = arr1D
    return out


def g_t(a, b):
    M = np.multiply(np.repeat(a, RANK, axis=0), np.repeat(b.T, RANK, axis=1))
    M -= np.diag(np.diag(M)/2)
    return M[np.triu_indices(RANK)] + M.T[np.triu_indices(RANK)]


def get_c(F):
    o = np.ones((2*F))
    z = np.zeros((F))
    return np.concatenate((o,z))

def get_g(M):
    F = M.shape[0]//2

    G = np.zeros((3*F, comb(RANK)))

    for i in range(F):
        G[i, :] = g_t(M[i], M[i])
        G[i+F, :] = g_t(M[i+F], M[i+F])
        G[i+(2*F), :] = g_t(M[i], M[i+F])

    return G

def sfm(W):

    a_f = np.mean(W, axis=1).reshape(W.shape[0], 1)
    print(W)
    W_aprox = np.matrix(W - a_f)
    print(W_aprox)

    # O = np.ones((W.shape[0]//2, W.shape[1]))

    # W_aprox = np.concatenate((W_aprox,O), axis=0)

    print(W_aprox)

    U, s, V = np.linalg.svd(W_aprox)

    U_ = U[:,:RANK]
    s_ = np.matrix(np.diag(s[:RANK]))
    V_ = V[:RANK,:]

    print(U_.shape)
    print(s_.shape)
    print(V_.shape)

    s_sqrt = sqrtm(s_)
    M_hat = U_
    S_hat = s_ * V_

    # Number of frames
    F = M_hat.shape[0]//2

    G = get_g(M_hat)
    c = get_c(F)

    l = lstsq(G, c)[0]

    L = np.matrix(squareform_diagfill(l))

    print(L)
    # A = cholesky(L)
    try:
        A = cholesky(L)
    except Exception as e:
        A = sqrtm(L)

    M = M_hat * A
    S = inv(A) * S_hat

    return M, S