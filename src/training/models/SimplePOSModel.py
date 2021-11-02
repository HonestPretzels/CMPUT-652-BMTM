# Based off of this tutorial https://nlpforhackers.io/lstm-pos-tagger-keras/
from consts import word_space_length, POS_space_length, sentence_max_length
from keras.models import Sequential
from keras.layers import Dense, LSTM, InputLayer, Bidirectional, TimeDistributed, Embedding, Activation

class POS_Model:
    '''
    This class contains the architecture for a keras model
    It also contains all the hyper-parameters necessary for that model
    '''

    def __init__(self):
        # TODO: Initialize the hyper-parameters
        self.optimizer = 'Adam'
        self.loss = 'categorical_crossentropy'
        self.learning_rate = 0.001
        self.batch_size = 128
        self.epochs = 50 # Lan et al. use 500 epochs for PTB data set
        self.validation_split = 0.2

        self.initModel()

    def initModel(self):
        self.model = Sequential()
        self.model.add(InputLayer((sentence_max_length,)))
        self.model.add(Embedding(word_space_length, 64)) # Is 64 only for PTB?
        self.model.add(Bidirectional(LSTM(256, return_sequences=True))) # should this be 380?
        self.model.add(TimeDistributed(Dense(POS_space_length)))
        self.model.add(Activation('softmax'))
        self.model.compile(loss=self.loss, optimizer=self.optimizer, metrics=['accuracy'])
        self.model.summary()

    def loadHyperParameters(self, config_dict):
        # TODO: Load the hyper parameters
        pass

    def loadCheckpoint(self, checkpoint):
        # TODO: Load a checkpoint into the model
        pass

    def saveCheckpoint(self, checkpoint):
        # TODO: Save a checkpoint
        pass

    def train(self, trainX, trainY):
        self.model.fit(trainX, trainY, batch_size=self.batch_size, epochs=self.epochs, validation_split=self.validation_split)
    
    def test(self, testX, testY, checkpoint=None):
        if checkpoint:
            self.loadCheckpoint(checkpoint)
        # TODO: Implement test code
        pass

    