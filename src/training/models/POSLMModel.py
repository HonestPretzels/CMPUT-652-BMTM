from models.SimplePOSModel import POS_Model
from models.SimpleLMModel import LM_Model

class POSLM_Model:
    model = None

    # this model is a combination the POS + LM models
    # output from POS is R --> input to LM
    # do we need to create a new class, or should we just define a function somewhere ...? unsure where

    # def __init__(self):

    # def initModel(self):

    # def loadCheckpoint(self, checkpoint):

    # def train(self, trainX, trainY):
