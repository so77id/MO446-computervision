
def query(queryDescriptors, distanceFunction, resultListSize, samplesDescriptors):
    results = []
    for trainId,trainSampleDescriptors in samplesDescriptors.items():
        d = dist(queryDescriptors, trainSampleDescriptors, distanceFunction)
        results.append((trainId,d))
    results.sort(key=lambda pair: pair[1], reverse=False)
    if len(results) > resultListSize:
        results = results[:resultListSize]
    return results

def dist(queryDescriptors, trainSampleDescriptors, distanceFunction):
    d = 0
    for queryDescriptor in queryDescriptors:
        dMin = 999999
        for trainSampleDescriptor in trainSampleDescriptors:
            d_ = distanceFunction(queryDescriptor, trainSampleDescriptor)
            if d_ < dMin:
                dMin = d_
        d += dMin
    return d