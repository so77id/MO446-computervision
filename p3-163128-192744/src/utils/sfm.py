from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import cv2
import numpy as np
from numpy.linalg import lstsq
from numpy.linalg import cholesky
from numpy.linalg import inv
from scipy.linalg import sqrtm
from scipy.linalg import norm


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


def g_t(a, b, RANK=4):
    M = np.multiply(np.repeat(a, RANK, axis=0), np.repeat(b.T, RANK, axis=1))
    M -= np.diag(np.diag(M)/2)
    return M[np.triu_indices(RANK)] + M.T[np.triu_indices(RANK)]


def get_c(F):
    o = np.ones((2*F))
    z = np.zeros((F))
    return np.concatenate((o,z))

def get_g(M, RANK=4):
    F = M.shape[0]//2

    G = np.zeros((3*F, comb(RANK)))

    for i in range(F):
        G[i, :] = g_t(M[i], M[i], RANK)
        G[i+F, :] = g_t(M[i+F], M[i+F], RANK)
        G[i+(2*F), :] = g_t(M[i], M[i+F], RANK)

    return G

def sfm(W, RANK=4):
    # Normalization
    a_f = np.mean(W, axis=1).reshape(W.shape[0], 1)
    W_aprox = np.matrix(W - a_f)

    # Only in homogeneous
    if RANK==4:
        O = np.ones((W.shape[0]//2, W.shape[1]))
        W_aprox = np.concatenate((W_aprox,O), axis=0)

    # SVD
    U, s, V = np.linalg.svd(W_aprox)

    U_ = U[:,:RANK]
    s_ = np.matrix(np.diag(s[:RANK]))
    V_ = V[:RANK,:]

    # Get hat variables
    M_hat = U_
    S_hat = s_ * V_

    # Number of frames
    F = M_hat.shape[0]//2

    # Get linear system
    G = get_g(M_hat, RANK)
    c = get_c(F)

    # Solve linear system
    l = lstsq(G, c)[0]

    # Create square matrix
    L = np.matrix(squareform_diagfill(l))

    # Cholesky
    A = cholesky(L)

    # Update
    M = M_hat * A
    S = inv(A) * S_hat

    return M, S

def get_camera_centers(M, RANK):

    F = M.shape[0] // 3

    C = np.matrix(np.ones((F, RANK)))

    for f in range(F):

        P = M[[f, f+F, f+2*F]]
        cr = -np.cross(M[f,:3],M[f + F,:3])
        C[f,:3] = cr / norm(cr)
        #C[f,:3] = -(inv(P[:,:3]) * (P[:,3])).flatten()

    return C
