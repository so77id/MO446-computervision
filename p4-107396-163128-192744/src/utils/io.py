from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pickle

def write(obj, output_file):
	with open(output_file, "wb") as f:
		pickle.dump(obj, f)

def read(file):
	with open(file, "rb") as f:
		return pickle.load(f)