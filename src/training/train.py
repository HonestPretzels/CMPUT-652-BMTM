import sys
import os
from SimplePOSModel import POS_Model
from SimpleLMModel import LM_Model
from POSLMModel import POSLM_Model
from consts import POS_space_length
from sklearn.model_selection import train_test_split
import tensorflow as tf
import numpy as np
import csv
import json
import re
# If you get an error that models are not included, run this script from the training directory

def getModel(modelType):
    '''
    Select and return model based on model type
    '''
    if modelType == '--POSLM':
        return POSLM_Model()
    elif modelType == '--LM':
        return LM_Model()
    else:
        return POS_Model()

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

def main():
    '''
    Called with command line arguments:
        1. The path to the dataset to be trained on. Will be split into train and test batch
        2. The path to the checkpoint directory
        3. The type of model to be trained (--POS, --LM, --POSLM)
        4. The hyperparameter config file
        5. if "--Tune" will do hyperparameter tuning

    '''
    dataPath = sys.argv[1]
    checkpointPath = sys.argv[2]
    modelType = sys.argv[3]
    if modelType == "--POS":
        print("Simple POS Model Selected")
        #TODO: Enable cross validation splits
        X = np.load(os.path.join(dataPath, 'PosX.npy'))
        Y = np.load(os.path.join(dataPath, 'PosY.npy'))
        xTrain,xTest, yTrain, yTest = train_test_split(X, Y)

    elif modelType == "--LM":
        print("Simple Language Model Selected")
        X = np.load(os.path.join(dataPath, 'Lm16to1X.npy'))
        Y = np.load(os.path.join(dataPath, 'Lm16to1Y.npy'))
        xTrain,xTest, yTrain, yTest = train_test_split(X, Y)

    elif modelType == "--POSLM":
        print("POS + LM Model Selected")
        X = np.load(os.path.join(dataPath, 'Lm16to1X.npy'))
        POSY = np.load(os.path.join(dataPath, 'PosY.npy'))
        LMY = np.load(os.path.join(dataPath, 'Lm16to1Y.npy'))
        xTrain, xTest, posYTrain, posYTest, lmYTrain, lmYTest = train_test_split(X, POSY, LMY)
        
    else:
        print("ERROR: Please Select a Valid Model Type")
        exit(1)


    model = getModel(modelType)
    if os.path.isdir(checkpointPath):
        if bool(os.listdir(checkpointPath)):
            model.loadCheckpoint(checkpointPath)
    else:
        os.mkdir(checkpointPath)

    if len(sys.argv) > 4:
        hyperParameterPath = sys.argv[4]
        setHyperParameters(model, hyperParameterPath)
        print("Hyper Parameters set based on %s"%hyperParameterPath)

    doTune = False
    if len(sys.argv) > 5:
        doTune = (sys.argv[5] == "--Tune")
        if doTune:
            print("Hyper Parameter Tuning Enabled")
    

    if doTune:
        # Set to 5 epochs to allow for quick grid search, do 5 epochs per hyperParameter set
        tuneHyperParameters(model, xTrain, yTrain, hyperParameterPath, 5)
    else:
        if modelType == "--POS":
            model.train(xTrain, tf.keras.utils.to_categorical(yTrain, POS_space_length), checkpointPath)
            model.saveCheckpoint(checkpointPath)
        elif modelType == "--LM":
            # TODO: Reshape this
            model.train(xTrain, yTrain, xTest, yTest)
            model.saveCheckpoint(checkpointPath)
        elif modelType == "--POSLM":
            posYTrain = tf.keras.utils.to_categorical(posYTrain, POS_space_length)
            posYTest = tf.keras.utils.to_categorical(posYTest, POS_space_length)
            model.train(xTrain, posYTrain, lmYTrain, xTest, posYTest, lmYTest, checkpointPath)
            


if __name__ == "__main__":
    main()