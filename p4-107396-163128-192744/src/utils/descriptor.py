from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import utils.features as ft
import utils.regions as reg

def get_description_function(descriptor_name="classic"):
    if descriptor_name == "classic":
        return ft.classic
    if descriptor_name == "lbp":
    	return ft.lbp

def get_descriptors(img, labeled_image, n_regions, descriptor_name="classic"):
	
    descriptor_function = get_description_function(descriptor_name)

    descriptors = []

    for i_region in range(n_regions):
        mask = labeled_image == i_region
        descriptors.append(descriptor_function(img, mask))

    return descriptors