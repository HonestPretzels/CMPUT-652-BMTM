from models.CLModel import CL_Model 
from models.SimplePOSModel import POS_Model 
from models.SimpleLMModel import LM_Model 
import numpy as np
import sys
import os
import json

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

def setHyperParameters(model, config):
    '''
    Set the hyper parameters of the model
    '''
    pass

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
        if not os.listdir(checkpointPath):
            initializeModel(model, checkpointPath)
    else:
        os.mkdir(checkpointPath)

    if len(sys.argv) > 4:
        hyperParameterPath = sys.argv[4]

    if len(sys.argv) > 5:
        doTune = (sys.argv[5] == "--Tune")
        print("Hyper Parameter Tuning Enabled")


if __name__ == "__main__":
    main()