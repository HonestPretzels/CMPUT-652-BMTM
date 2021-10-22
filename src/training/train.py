import sys
from models.CLModel import CL_Model 
from models.SimplePOSModel import POS_Model 
from models.SimpleLMModel import LM_Model 
import numpy as np
import os
import csv
import json

# If you get an error that models are not included, run this script from the training directory

def getModel(modelType):
    '''
    Select and return model based on model type
    '''
    
    pass

def initializeModel(model, checkpoint):
    '''
    If the model has been pretrained load those weights in
    '''
    pass

def trainModel(model, trainingData, validationData, checkPointPath, numEpochs):
    '''
    Train the model and save the weights to the checkpointPath
    '''
    pass

def tuneHyperParameters(model, trainingData, validationData, outputPath, numEpochs):
    '''
    Do some form of hyper parameter tuning here and output a JSON to the outputPath
    '''
    pass

def setHyperParameters(model, configPath):
    '''
    Set the hyper parameters of the model
    '''
    pass

def getDataSet(p):
    '''
    Open the files at the path and load the data
    TODO: Enable loading separate files for train and validation
    TODO: Decide how to handle sentence breaks
    TODO: Enable cross validation splits with some sort of offset marker
    TODO: Convert to word vectors
    TODO: Apply normalization to all data
    '''
    training = []
    validation = []
    extension = os.path.splitext(p)[1]
    if extension == ".csv":
        x = []
        y = []
        with open(p, "r", newline="") as f:
            reader = csv.reader(f)
            for line in reader:
                x.append(line[0])
                y.append(line[1])

    elif extension == ".txt":
        x = []
        y = []
        with open(p, "r") as f:
            for line in f:
                if line == "\n":
                    continue
                a = line.split(' ')[0].strip()
                b = line.split(' ')[1].strip()
                x.append(a)
                y.append(b)
    else:
        print("ERROR: Unknown file type for dataset. Please use txt or csv files")
        exit(1)

    print(y)

    xTrain = x[:len(x)*0.9]
    yTrain = y[:len(y)*0.9]
    xValid = x[len(x)*0.9:]
    yValid = y[len(y)*0.9:]
    trainSet = [(xTrain[i], yTrain[i]) for i in range(len(xTrain))]
    validSet = [(xValid[i], yValid[i]) for i in range(len(xValid))]


    return trainSet, validSet

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
    if modelType == "--POS":
        print("Simple POS Model Selected")
    elif modelType == "--LM":
        print("Simple Language Model Seleceted")
    elif modelType == "--CL":
        print("Continuous Learning Model Selected")
    else:
        print("ERROR: Please Select a Valid Model Type")
        exit(1)

    model = getModel(modelType)
    if os.path.isdir(checkpointPath):
        if bool(os.listdir(checkpointPath)):
            initializeModel(model, checkpointPath)
    else:
        os.mkdir(checkpointPath)

    if len(sys.argv) > 4:
        hyperParameterPath = sys.argv[4]
        setHyperParameters(model, hyperParameterPath)
        print("Hyper Parameters set based on %s"%hyperParameterPath)


    if len(sys.argv) > 5:
        doTune = (sys.argv[5] == "--Tune")
        if doTune:
            print("Hyper Parameter Tuning Enabled")

    xTrain, xValid, yTrain, yValid= getDataSet(dataPath)


if __name__ == "__main__":
    main()