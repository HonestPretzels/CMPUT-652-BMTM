# TODO: identify the vocab length, add it into the consts script, then here
from consts import word_space_length, sentence_max_length # , vocab_length
from keras.models import Sequential
from keras.layers import Dense, LSTM, InputLayer, Bidirectional,  TimeDistributed, Embedding, Activation


class LM_Model:

    model = None

    def __init__(self):
        self.optimizer = "Adam"
        self.loss = 'categorical_crossentropy'
        self.learning_rate = 0.001
        self.batch_size = 128
        self.epochs = 50 # Lan et al. use 500 epochs for PTB data set
        self.validation_split = 0.2

        self.initModel()

    def initModel(self):
        model = None
        self.model = Sequential()
        self.model.add(InputLayer((sentence_max_length,)))
        self.model.add(Embedding(word_space_length, 64))  # Is 64 only for PTB?

        # TODO: Add the output from the POS model to the first hidden layer
        self.model.add(Bidirectional(LSTM(380, return_sequences=True)))

        self.model.add(Bidirectional(LSTM(380, return_sequences=True)))
        self.model.add(Bidirectional(LSTM(380, return_sequences=True)))

        # TODO: ensure vocab_length is imported into this file
        self.model.add(TimeDistributed(Dense(vocab_length)))
        self.model.add(Activation('softmax'))

        # TODO: Add the perplexity metric - should we do this using a package or write the fcn. ourselves?
        # using accuracy as metric as a placeholder so that the code runs
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
<<<<<<< HEAD
        self.model.fit(trainX, trainY, batch_size=self.batch_size, epochs=self.epochs,
                       validation_split=self.validation_split)

    def test(self, testX, testY, checkpoint=None):
        if checkpoint:
            self.loadCheckpoint(checkpoint)
        # TODO: Implement test code
        pass
=======
        self.model.fit(trainX, trainY, batch_size=self.batch_size, epochs=self.epochs, validation_split=self.validation_split)
    
    def test(self, testX, testY):
        self.model.evaluate(testX, testY, batch_size=self.batch_size)
        # TODO: Fix metrics to show more than just accuracy
>>>>>>> origin/HarryPotterData
