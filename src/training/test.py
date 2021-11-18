from train import getModel, getDataSet, to_categorical
from consts import POS_space_length
import sys

def main():
    dataPath = sys.argv[1]
    checkpointPath = sys.argv[2]
    modelType = sys.argv[3]
    wordPath = sys.argv[4]
    posPath = sys.argv[5]
    pass

    xTest, yTest, _, _ = getDataSet(dataPath, wordPath, posPath)

    model = getModel(modelType)
    hyp_dict = {"batch_size": 48, "oogaBooga":"ahhhhhh"}
    model.loadHyperParameters(hyp_dict)
    # model.loadCheckpoint(checkpointPath)
    # model.test(xTest, to_categorical(yTest, POS_space_length))

if __name__ == "__main__":
    main()