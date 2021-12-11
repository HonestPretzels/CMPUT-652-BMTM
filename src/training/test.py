from train import getModel
import numpy as np
import tensorflow as tf
from consts import POS_space_length
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
    if modelType == "--POS" or modelType == "--POSFC":
        Y = tf.keras.utils.to_categorical(Y, POS_space_length)
    model.test(X, Y)

if __name__ == "__main__":
    main()