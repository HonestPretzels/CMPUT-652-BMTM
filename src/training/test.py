from train import getModel
import numpy as np
import os
import sys

def main():
    dataPath = sys.argv[1]
    checkpointPath = sys.argv[2]
    output = sys.argv[3]
    modelType = sys.argv[4]

    model = getModel(modelType)
    model.goToHiddenRep()
    model.loadCheckpoint(checkpointPath)
    X = np.load(dataPath)
    hiddenReps = model.predict(X)
    hiddenReps = hiddenReps.reshape((hiddenReps.shape[0], hiddenReps.shape[1]*hiddenReps.shape[2]))
    print(hiddenReps.shape)
    np.save(output, hiddenReps)

if __name__ == "__main__":
    main()