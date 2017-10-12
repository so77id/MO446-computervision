import numpy as np
from scipy.spatial import distance
from collections import OrderedDict
import util, descriptor, search

#loads the samples, and returns a hash table <id,sample_dict>
def loadSamples():
    filePaths = util.getDirFilePaths('input')
    samplesByID = OrderedDict()
    for filePath in filePaths:
        name = filePath.rsplit('/')[1].rsplit('.')[0]
        label,index = name.rsplit('_')
        samplesByID[name] = (dict(id=name, path=filePath, label=label, index=index))
    return samplesByID

if __name__ == '__main__':
    #CONFIGS:
    numSegments = 5  # K for K-means
    descriptorsCacheDir = 'output/descriptors'
    distanceFunctions = [distance.euclidean, distance.correlation]
    resultSize = 3
    queryIDs = ['beach_2', 'boat_5', 'cherry_3', 'pond_2', 'stHelens_2', 'sunset1_2', 'sunset2_2']

    samplesByID = loadSamples()

    samplesDescriptors = descriptor.extractDescriptors(samplesByID.values(), descriptorsCacheDir, numSegments)
    for distanceFunction in distanceFunctions:
        precisionsAt, averagePrecisionsAt = [], []
        for queryID in queryIDs:
            queryLabel = samplesByID[queryID]['label']
            queryDescriptors = samplesDescriptors[queryID]
            searchResult = search.query(queryDescriptors, distanceFunction, resultSize, samplesDescriptors)
            predictedLabels = [samplesByID[r['id']]['label'] for r in searchResult]
            print('results for query '+queryID+' with distance '+ distanceFunction.__name__ +': ' + str(searchResult))
            #computes quality:
            precisionAt, averagePrecisionAt = search.precision(predictedLabels, queryLabel), search.averagePrecision(predictedLabels, queryLabel)
            precisionsAt.append(precisionAt)
            averagePrecisionsAt.append(averagePrecisionAt)
            print('\tP@%d: %f, AP@%d: %f' % (resultSize, precisionAt, resultSize, averagePrecisionAt))
        meanP, stdP = np.mean(precisionsAt), np.std(precisionsAt)
        meanAP, stdAP = np.mean(averagePrecisionsAt), np.std(averagePrecisionsAt)
        print('# results for distance %s: P@%d: %.2f +- %.2f, AP@%d: %.2f +- %.2f' %
            (distanceFunction.__name__, len(queryIDs), meanP*100, stdP*100, len(queryIDs), meanAP*100, stdAP*100))
