
def query(queryDescriptors, distanceFunction, resultListSize, samplesDescriptors):
    results = []
    for trainId,trainSampleDescriptors in samplesDescriptors.items():
        d = dist(queryDescriptors, trainSampleDescriptors, distanceFunction)
        results.append(dict(id=trainId,distance=d))
    results.sort(key=lambda pair: pair['distance'], reverse=False)
    if len(results) > resultListSize:
        results = results[:resultListSize]
    return results

def dist(queryDescriptors, trainSampleDescriptors, distanceFunction):
    #TODO mocking distances for now, while descriptors are not done yet
    import random
    return round(random.uniform(0.1,1.0),2)

    d = 0
    for queryDescriptor in queryDescriptors:
        dMin = 999999
        for trainSampleDescriptor in trainSampleDescriptors:
            d_ = distanceFunction(queryDescriptor, trainSampleDescriptor)
            if d_ < dMin:
                dMin = d_
        d += dMin
    return d

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
