from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import cv2
import numpy as np
from scipy.spatial import distance

def queries(distanceName, resultListSize, samplesDescriptors, queryIDs):
	f = getattr(distance, distanceName)
	return [{'q':queryID, 'r':query(f, resultListSize, samplesDescriptors, queryID)} for queryID in queryIDs]

def query(distanceFunction, resultListSize, samplesDescriptors, queryID):
	queryDescriptor = samplesDescriptors[queryID]
	results = []
	for trainID,trainSampleDescriptor in samplesDescriptors.items():
		if queryID != trainID:
			d = distanceFunction(queryDescriptor, trainSampleDescriptor)
			results.append(dict(id=trainID,distance=d))
	results.sort(key=lambda pair: pair['distance'], reverse=False)
	if len(results) > resultListSize:
		results = results[:resultListSize]
	return results

def evaluate_results(searchs_results, labelsByID):
	precisions, averagePrecisions = [], []
	for result in searchs_results:
		query_id, searchResult = result['q'], result['r']
		query_class = labelsByID[query_id]
		predictedClasses = [labelsByID[r['id']] for r in searchResult]
		p = precision(predictedClasses, query_class)
		ap = averagePrecision(predictedClasses, query_class)
		precisions.append(p), averagePrecisions.append(ap)
	return precisions, averagePrecisions

def precision(predictedClasses, groundTruthClass):
	return predictedClasses.count(groundTruthClass) / len(predictedClasses)

#calculates AP@K, assuming there are more possible correct results than those obtained
def averagePrecision(predictedClasses, groundTruthClass):
	score = 0.0
	hits = 0.0
	for i,p in enumerate(predictedClasses):
		if p == groundTruthClass:
			hits += 1.0
			score += hits / (i + 1.0)
	return score / len(predictedClasses)

def format_mean_std(values):
	mean, std = np.mean(values), np.std(values)
	return '%.2f +- %.2f' % (mean*100, std*100)

def print_results(searchs_results, precisions, averagePrecisions, p_formatted, ap_formatted, img_path_expression, output_file):
	nQueries = len(searchs_results)
	resultSize = len(searchs_results[0]['r'])
	cols,rows = 160,120
	border_id = 50
	margin_id = 10
	x_off = cols*(resultSize+1) + 10
	y_off = (rows+border_id)*nQueries
	spaces_color = (0,0,0)
	text_color = (0,0,255)
	font = cv2.FONT_HERSHEY_SIMPLEX
	fontSize = 0.5
	img_results = None
	for index,search_result in enumerate(searchs_results):
		query_id = search_result['q']
		ids = [query_id] + [r['id'] for r in search_result['r']]
		img_result = None
		for id in ids:
			img = cv2.resize(cv2.imread(img_path_expression % id), (cols, rows))
			img = cv2.copyMakeBorder(img, 0,border_id,0,0, cv2.BORDER_CONSTANT, spaces_color)
			cv2.putText(img, str(id), (margin_id,rows+20), font, fontSize, text_color)
			img_result = img if img_result is None else np.concatenate((img_result, img), axis=1)
		img_result = cv2.copyMakeBorder(img_result, 0,0,0,190, cv2.BORDER_CONSTANT, spaces_color)
		cv2.putText(img_result, ' P: %.2f'% (precisions[index]       *100), (x_off,rows//2   ), font, fontSize, text_color)
		cv2.putText(img_result, 'AP: %.2f'% (averagePrecisions[index]*100), (x_off,rows//2+20), font, fontSize, text_color)
		img_results = img_result if img_results is None else np.concatenate((img_results,img_result), axis=0)
	
	img_results = cv2.copyMakeBorder(img_results, 0,40,0,0, cv2.BORDER_CONSTANT, spaces_color)
	cv2.putText(img_results, ' P: '+ p_formatted, (x_off,y_off   ), font, fontSize, text_color)
	cv2.putText(img_results, 'AP: '+ap_formatted, (x_off,y_off+20), font, fontSize, text_color)
	cv2.imwrite(output_file, img_results)
