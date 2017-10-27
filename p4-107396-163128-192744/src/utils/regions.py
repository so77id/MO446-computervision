from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import cv2
import numpy as np
import scipy.stats as st
from collections import deque

def segment_image(img, n_clusters=5, blur = 0, preprocess = False):

    if(blur > 0):
        img = cv2.GaussianBlur(img, (blur,blur),0)

    if preprocess:
        img = (img - img.min(axis=(0,1)))  #avoid negative values that can not be visualized in opencv
        img = ((img/img.max(axis=(0,1)) )* 255).astype(np.uint8)

    data = img.reshape((-1,3))

    data = np.float32(data)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)

    ret,label,center = cv2.kmeans(data, n_clusters, None, criteria, 10, cv2.KMEANS_PP_CENTERS)

    return label.reshape((img.shape[:2]))

def get_neighborhood(shape, x, y):
    neigh = []
    for i in range(-1,2):
        for j in range(-1,2):
            if 0 <= x + i < shape[0] and 0 <= y+j < shape[1]:
                neigh.append([x + i, y + j])

    return neigh

def dfs(mask, x, y):
    dfs_mask = np.zeros_like(mask)

    dfs_mask[x,y] = 1
    queue = deque([[x,y]])

    while queue:
        x,y = queue.popleft()

        neigh = get_neighborhood(mask.shape, x, y)
        for n in neigh:
            if mask[n[0], n[1]] == 1 and dfs_mask[n[0], n[1]] == 0:
                dfs_mask[n[0], n[1]] = 1
                queue.append(n)

    return dfs_mask


def connected_components(img, n_clusters):
    new_img = np.zeros_like(img)
    k = 0

    for i in range(n_clusters):
        mask_b = img==i
        mask_i = mask_b.astype(np.int)

        while mask_i.sum() > 0:
            inds = np.where(mask_i == 1)
            x = inds[0][0]
            y = inds[1][0]

            dfs_mask = dfs(mask_i, x, y)

            new_img[dfs_mask==1] = k
            mask_i -= dfs_mask
            k+=1

    return new_img, k


def get_regions(img, n_clusters=5):
    kmeans_regions = segment_image(img, n_clusters)
    connected_regions, k = connected_components(kmeans_regions, n_clusters)

    return connected_regions, k
