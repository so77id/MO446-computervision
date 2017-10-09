import os, pickle
import util

def extractDescriptors(samples, descriptorsCacheDir, numSegments):
    samplesDescriptors = dict()
    for sample in samples:
        id = sample['id']
        sampleDescriptorsFile = os.path.join(descriptorsCacheDir, str(id))
        if os.path.isfile(sampleDescriptorsFile) is False:
            segments = segment(sample, numSegments)
            descriptors = describeSegments(segments)
            util.createDirIfAbsent(descriptorsCacheDir)
            with open(sampleDescriptorsFile, "wb") as f:
                pickle.dump(descriptors, f)
        else:
            with open(sampleDescriptorsFile, "rb") as f:
                descriptors = pickle.load(f)
        samplesDescriptors[id] = descriptors
    return samplesDescriptors

def segment(sample, numSegments):
    #TODO
    return []

def describeSegments(segments):
    descriptors = []
    for segment in segments:
        descriptor = []  # TODO
        descriptors.append(descriptor)
    return descriptors