import sys
import os
from SimplePOSModel import POS_Model
from SimpleLMModel import LM_Model
from FCLMModel import FC_LM_Model
from FCPOSModel import FC_POS_Model
from POSLMModel import POSLM_Model
from MajorityGuessModel import TrivialModel
from consts import POS_space_length
from sklearn.model_selection import train_test_split
import tensorflow as tf
import numpy as np

# If you get an error that models are not included, run this script from the training directory

def getModel(modelType):
    '''
    Select and return model based on model type
    '''
    if modelType == '--POSLM':
        return POSLM_Model()
    elif modelType == '--LM':
        return LM_Model()
    elif modelType == "--POS":
        return POS_Model()
    elif modelType == "--POSFC":
        return FC_POS_Model()
    elif modelType == "--LMFC":
        return FC_LM_Model()
    else:
        return TrivialModel()


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
    if modelType == "--POS" or modelType == "--POSFC" or modelType == "--POSG":
        print("POS Model Selected")
        #TODO: Enable cross validation splits
        X = np.load(os.path.join(dataPath, 'PosX_train.npy'))
        Y = np.load(os.path.join(dataPath, 'PosY_train.npy'))
        xTrain, xVal, yTrain, yVal = train_test_split(X, Y)

    elif modelType == "--LM" or modelType == "--LMFC" or modelType == "--LMG":
        print("Language Model Selected")
        X = np.load(os.path.join(dataPath, 'Lm16to1X_train.npy'))
        Y = np.load(os.path.join(dataPath, 'Lm16to1Y_train.npy'))
        xTrain, xVal, yTrain, yVal = train_test_split(X, Y)

    elif modelType == "--POSLM":
        print("POS + LM Model Selected")
        X = np.load(os.path.join(dataPath, 'Lm16to1X_train.npy'))
        POSY = np.load(os.path.join(dataPath, 'PosY_train.npy'))
        LMY = np.load(os.path.join(dataPath, 'Lm16to1Y_train.npy'))
        xTrain, xVal, posYTrain, posYVal, lmYTrain, lmYVal = train_test_split(X, POSY, LMY)
        
    else:
        print("ERROR: Please Select a Valid Model Type")
        exit(1)


    model = getModel(modelType)

    doTune = False
    if len(sys.argv) > 4:
        doTune = (sys.argv[4] == "--Tune")
        if doTune:
            print("Hyper Parameter Tuning Enabled")
    

    if doTune:
        # Set to 5 epochs to allow for quick grid search, do 5 epochs per hyperParameter set
        batchSizes = [50,100,1000]
        learningRates = [0.1,0.01, 0.001]
        for b in batchSizes:
            for l in learningRates:
                print("Batch Size:", b, "Learning Rate:", l)
                model.loadHyperParameters({'batch_size':b, 'learning_rate': l, 'epochs': 5})
                model.initModel()
                checkpointPath = os.path.join(checkpointPath, 'HyperParamaters_b_%d_lr_%d'%(b,learningRates.index(l)))
                if not os.path.isdir(checkpointPath):
                    os.mkdir(checkpointPath)
                if modelType == "--POS" or modelType == "--POSFC":
                    posYTrain = tf.keras.utils.to_categorical(yTrain, POS_space_length)
                    posYVal = tf.keras.utils.to_categorical(yVal, POS_space_length)
                    model.train(xTrain, posYTrain, xVal, posYVal, checkpointPath)
                    model.saveCheckpoint(checkpointPath)
                elif modelType == "--LM" or modelType == "--LMFC":
                    # TODO: Reshape this
                    model.train(xTrain, yTrain, xVal, yVal, checkpointPath)
                    model.saveCheckpoint(checkpointPath)
                elif modelType == "--POSLM":
                    posYTrain = tf.keras.utils.to_categorical(posYTrain, POS_space_length)
                    posYVal = tf.keras.utils.to_categorical(posYVal, POS_space_length)
                    model.train(xTrain, posYTrain, lmYTrain, xVal, posYVal, lmYVal, checkpointPath)
    else:           
        if modelType == "--POS" or modelType == "--POSFC":
            posYTrain = tf.keras.utils.to_categorical(yTrain, POS_space_length)
            posYVal = tf.keras.utils.to_categorical(yVal, POS_space_length)
            model.train(xTrain, posYTrain, xVal, posYVal, checkpointPath)
            model.saveCheckpoint(checkpointPath)
        elif modelType == "--LM" or modelType == "--LMFC":
            # TODO: Reshape this
            model.train(xTrain, yTrain, xVal, yVal, checkpointPath)
            model.saveCheckpoint(checkpointPath)
        elif modelType == "--POSLM":
            posYTrain = tf.keras.utils.to_categorical(posYTrain, POS_space_length)
            posYVal = tf.keras.utils.to_categorical(posYVal, POS_space_length)
            model.train(xTrain, posYTrain, lmYTrain, xVal, posYVal, lmYVal, checkpointPath)
        elif modelType == "--POSG":
            model.train(xTrain, yTrain)
            preds = model.predict(xVal).flatten()
            yVal = yVal.flatten()
            accuracy = (yVal == preds).sum() / preds.shape[0]
            print("accuracy with majority guess is: ", accuracy)
        elif modelType == "--LMG":
            model.train(xTrain, yTrain)
            preds = model.predict(xVal)
            yVal = yVal.flatten()
            accuracy = (yVal == preds).sum() / preds.shape[0]
            print("accuracy with majority guess is: ", accuracy)
            


if __name__ == "__main__":
    main()