from scipy.spatial import distance
import util, descriptor, search

def loadSamples():
    filePaths = util.getDirFilePaths('input')
    samples = []
    for filePath in filePaths:
        name = filePath.rsplit('/')[1].rsplit('.')[0]
        label,index = name.rsplit('_')
        samples.append(dict(id=name, path=filePath, label=label, index=index))
    return samples

def getSample(samples, id):
    return next(s for s in samples if s['id'] == id)

if __name__ == '__main__':
    #CONFIGS:
    numSegments = 5  # K for K-means
    descriptorsCacheDir = 'output/descriptors'
    distanceFunctions = [distance.euclidean, distance.correlation]
    resultListSize = 3
    queryIDs = ['beach_2', 'boat_5', 'cherry_3', 'pond_2', 'stHelens_2', 'sunset1_2', 'sunset2_2']

    samples = loadSamples()

    samplesDescriptors = descriptor.extractDescriptors(samples, descriptorsCacheDir, numSegments)
    for queryID in queryIDs:
        querySample = getSample(samples, queryID)
        queryDescriptors = samplesDescriptors[querySample['id']]
        for distanceFunction in distanceFunctions:
            searchResult = search.query(queryDescriptors, distanceFunction, resultListSize, samplesDescriptors)
            print('results for query '+queryID+' with distance '+ distanceFunction.__name__ +': ' + str(searchResult))
