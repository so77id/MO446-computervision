import os, pathlib

# Loads the files within the dir, in alphabetical order
def getDirFilePaths(dir):
    names = os.listdir(dir)
    names.sort()
    return [os.path.join(dir, name) for name in names]

def createDirIfAbsent(path):
    pathlib.Path(path).mkdir(parents=True, exist_ok=True)  # creates the output folder, if necessary
