from train import getModel
import numpy as np
import os
import sys

def main():
    dataPath = sys.argv[1]
    targetPath = sys.argv[2]
    modelType = sys.argv[3]
    checkpointPath = sys.argv[4]

    model = getModel(modelType)

    model.loadCheckpoint(checkpointPath)
    X = np.load(dataPath)
    Y = np.load(targetPath)
    model.test(X, Y)

if __name__ == "__main__":
    main()