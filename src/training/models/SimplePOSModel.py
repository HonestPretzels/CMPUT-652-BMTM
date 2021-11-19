# Based off of this tutorial https://nlpforhackers.io/lstm-pos-tagger-keras/
from consts import word_space_length, POS_space_length, sentence_max_length
from keras.models import Sequential, load_model
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
        self.sentence_max = sentence_max_length
        self.POS_space = POS_space_length
        self.word_space = word_space_length
        self.lstm_dropout = 0.8

        self.initModel()

    def initModel(self):
        self.model = Sequential()
        self.model.add(InputLayer((self.sentence_max,)))
        self.model.add(Embedding(self.word_space, 64))
        self.model.add(Bidirectional(LSTM(256, dropout=self.lstm_dropout, return_sequences=True)))
        self.model.add(TimeDistributed(Dense(self.POS_space)))
        self.model.add(Activation('softmax'))
        self.model.compile(loss=self.loss, optimizer=self.optimizer, metrics=['accuracy'])
        self.model.summary()

    def loadHyperParameters(self, config_dict):
        for key in config_dict:
            try:
                getattr(self, key)
                setattr(self, key, config_dict[key])
            except:
                continue

    def loadCheckpoint(self, checkpoint):
        print('Loading Checkpoint: %s'%checkpoint)
        self.model = load_model(checkpoint)

    def saveCheckpoint(self, checkpoint):
        self.model.save(checkpoint)

    def train(self, trainX, trainY):
        self.model.fit(trainX, trainY, batch_size=self.batch_size, epochs=self.epochs, validation_split=self.validation_split)
    
    def test(self, testX, testY):
        self.model.evaluate(testX, testY, batch_size=self.batch_size)
        # TODO: Fix metrics to show more than just accuracy

    