from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import argparse
import utils.io as io
import utils.search as search

def main(argv):
	parser = argparse.ArgumentParser()
	parser.add_argument('-d', '--descriptorsFile', help='descriptors dict file', required=True)
	parser.add_argument('-q', '--queryIDs', help='query IDs joined by + symbol', required=True)
	parser.add_argument('-s', '--resultSize', help='result list size', required=True)
	parser.add_argument('-f', '--distanceName', help='distance function name', required=True)
	parser.add_argument('-i', '--img_path_expression', help='the img path expression in which %s is the ID', required=True)
	parser.add_argument('-o', '--plot_file', help='the output file for plotting', required=True)
	ARGS = parser.parse_args()
	samplesDescriptors = io.read(ARGS.descriptorsFile)
	queryIDs = ARGS.queryIDs.split('+')
	resultSize = int(ARGS.resultSize)
	distanceName = ARGS.distanceName
	img_path_expression = ARGS.img_path_expression
	plot_file = ARGS.plot_file

	labelsByID = dict()
	for id in samplesDescriptors.keys():
		labelsByID[id] = id.rsplit('_')[0]

	searchs_results = search.queries(distanceName, resultSize, samplesDescriptors, queryIDs)
	for r in searchs_results:
		print(r)
	precisions, averagePrecisions = search.evaluate_results(searchs_results, labelsByID)
	p_formatted = search.format_mean_std(precisions)
	ap_formatted = search.format_mean_std(averagePrecisions)
	print('results using distance %s:' % distanceName)
	print('P@%d by query:' % resultSize, precisions)
	print('AP@%d by query:' % resultSize, averagePrecisions)
	print('P@%d: %s, AP@%d: %s' % (resultSize, p_formatted, resultSize, ap_formatted))

	search.print_results(searchs_results, precisions, averagePrecisions, p_formatted, ap_formatted, img_path_expression, plot_file)
	
if __name__ == "__main__":
	sys.exit(main(sys.argv[1:]))