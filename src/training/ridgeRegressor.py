from numpy.core.fromnumeric import mean
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
import numpy as np
import sys
from tqdm import tqdm

def main():
    Xpath = sys.argv[1]
    Ypath = sys.argv[2]
    resultsPath = sys.argv[3]
    
    
    X = np.load(Xpath)
    Y = np.load(Ypath)
    numVoxels = Y.shape[1]
    
    trainX, testX, trainY, testY = train_test_split(X, Y, test_size=0.1)
    
    scores = np.zeros((numVoxels))
    
    for i in tqdm(range(numVoxels)):
        currTrainY = trainY[:,i]
        currTestY = testY[:,i]
        model = Ridge(alpha=1.0)
        model.fit(trainX, currTrainY)
        score = model.score(testX, currTestY)
        scores[i] = score
    np.save(resultsPath, scores)

if __name__ == "__main__":
    main()