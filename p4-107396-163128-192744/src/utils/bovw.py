from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import cv2
import numpy as np

def get_dictionary(descriptors, n_features):
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)

    data = []
    for value in descriptors.values():
        data.extend(value)

    data = np.float32(data)

    data = np.nan_to_num(data)  # handles NaN, inf and -inf

    #reescale features to [0,1], so that each one has equal importance:
    data -= data.min(axis=0)
    data /= data.max(axis=0)

    _, labels, centers = cv2.kmeans(data, n_features, None, criteria, 10, cv2.KMEANS_PP_CENTERS)
    return labels, centers

def create_descriptors(labels, ids_descriptors, n_features):
    cont = 0
    images_descriptors = {}
    for key, descriptors in ids_descriptors.items():
        descriptor = np.zeros((n_features,1))
        for des in descriptors:
            descriptor[labels[cont]]+=1
            cont+=1

        images_descriptors[key] = descriptor

    return images_descriptors

def describe_images(ids_descriptors, n_features):
    labels, visualwords = get_dictionary(ids_descriptors, n_features)
    images_descriptors = create_descriptors(labels, ids_descriptors, n_features)
    return images_descriptors, visualwords
