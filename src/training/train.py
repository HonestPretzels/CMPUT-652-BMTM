import sys
from models.CLModel import CL_Model 
from models.SimplePOSModel import POS_Model 
from models.SimpleLMModel import LM_Model 
from consts import sentence_max_length, POS_space_length
from keras.preprocessing.sequence import pad_sequences
import numpy as np
import os
import csv
import json
import re

# If you get an error that models are not included, run this script from the training directory

def getModel(modelType):
    '''
    Select and return model based on model type
    '''
    if modelType == '--CL':
        return CL_Model()
    elif modelType == '--LM':
        return LM_Model()
    else:
        return POS_Model()

def to_categorical(sequences, categories):
    '''
    One-hotify a sequence. Taken from here: https://nlpforhackers.io/lstm-pos-tagger-keras/
    '''
    cat_sequences = []
    for s in sequences:
        cats = []
        for item in s:
            cats.append(np.zeros(categories))
            cats[-1][item] = 1.0
        cat_sequences.append(cats)
    return np.array(cat_sequences)

def tuneHyperParameters(model, trainX, trainY):
    '''
    Do some form of hyper parameter tuning here and output a JSON to the outputPath
    '''
    pass

def setHyperParameters(model, configPath):
    '''
    Set the hyper parameters of the model
    '''
    with open(configPath, 'r') as f:
        configDict = json.loads(f)
        model.loadHyperParameters(configDict)

def vectorize(sequence, dataFile):
    '''
    Taking a sequence of words or tags, return a N*d np array of one-hot vectors
    where N is the number of items in the list and d is the length of the
    vector space
    '''
    with open(dataFile, 'r') as f:
        lines = f.readlines()
        tags = [line.split(':')[0] for line in lines]
    intSequences = []
    
    for i in range(len(sequence)):
        intSequences.append([tags.index(re.sub(r'[^\w\s]', '', item)) for item in sequence[i]])
    return pad_sequences(intSequences, maxlen=sentence_max_length, padding="post")

def getDataSet(p, wp, posp):
    '''
    Open the files at the path and load the data
    TODO: Enable loading separate files for train and validation
    TODO: MAKE HARRY POTTER DATA WORK
    '''
    extension = os.path.splitext(p)[1]
    if extension == ".csv":
        x = []
        y = []
        with open(p, "r", newline="") as f:
            currX = []
            currY = []
            reader = csv.reader(f)
            for line in reader:
                if len(currX) == 4:
                    x.append(currX)
                    y.append(currY)
                    currX = []
                    currY = []
                currX.append(line[0])
                currY.append(line[1])

    elif extension == ".txt":
        x = []
        y = []
        with open(p, "r") as f:
            currX = []
            currY = []
            for line in f:
                if line == "\n":
                    x.append(currX)
                    y.append(currY)
                    currX = []
                    currY = []
                    continue
                if len(currX) == 4:
                    x.append(currX)
                    y.append(currY)
                    currX = []
                    currY = []
                    
                a = line.split(' ')[0].strip()
                b = line.split(' ')[1].strip()
                currX.append(a)
                currY.append(b)
    else:
        print("ERROR: Unknown file type for dataset. Please use txt or csv files")
        exit(1)

    x = vectorize(x, wp)
    y = vectorize(y, posp)

    xTrain = x[:round(len(x)*0.9)]
    yTrain = y[:round(len(y)*0.9)]
    xValid = x[round(len(x)*0.9):]
    yValid = y[round(len(y)*0.9):]


    return xTrain, yTrain, xValid, yValid

def main():
    '''
    Called with command line arguments:
        1. The path to the dataset to be trained on. Will be split into train and test batch
        2. The path to the checkpoint directory
        3. The type of model to be trained (--POS, --LM, --CL)
        4. The hyperparameter config file
        4. if "--Tune" will do hyperparameter tuning

    '''
    dataPath = sys.argv[1]
    checkpointPath = sys.argv[2]
    modelType = sys.argv[3]
    wordPath = sys.argv[4]
    posPath = sys.argv[5]
    if modelType == "--POS":
        print("Simple POS Model Selected")
    elif modelType == "--LM":
        print("Simple Language Model Seleceted")
    elif modelType == "--CL":
        print("Continuous Learning Model Selected")
    else:
        print("ERROR: Please Select a Valid Model Type")
        exit(1)

    #TODO: Enable cross validation splits
    xTrain, yTrain, xTest, yTest = getDataSet(dataPath, wordPath, posPath)

    model = getModel(modelType)
    if os.path.isdir(checkpointPath):
        if bool(os.listdir(checkpointPath)):
            model.loadCheckpoint(checkpointPath)
    else:
        os.mkdir(checkpointPath)

    if len(sys.argv) > 6:
        hyperParameterPath = sys.argv[6]
        setHyperParameters(model, hyperParameterPath)
        print("Hyper Parameters set based on %s"%hyperParameterPath)

    doTune = False
    if len(sys.argv) > 7:
        doTune = (sys.argv[7] == "--Tune")
        if doTune:
            print("Hyper Parameter Tuning Enabled")
    

    if doTune:
        # Set to 5 epochs to allow for quick grid search, do 5 epochs per hyperParameter set
        tuneHyperParameters(model, xTrain, yTrain, hyperParameterPath, 5)
    else:
        model.train(xTrain, to_categorical(yTrain, POS_space_length))
        model.saveCheckpoint(checkpointPath)


if __name__ == "__main__":
    main()