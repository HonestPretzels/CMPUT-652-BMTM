from consts import word_space_length, POS_space_length, sentence_max_length
from keras.models import Sequential, load_model
from keras.layers import Dense, LSTM, InputLayer, Bidirectional, \
    TimeDistributed, Embedding, Activation, Concatenate


class POSLM_Model:

    model = None

    def __init__(self):
        self.optimizer = 'Adam'
        self.loss = 'categorical_crossentropy'
        self.learning_rate = 0.001
        self.batch_size = 128
        self.epochs = 50
        self.validation_split = 0.2
        self.sentence_max = sentence_max_length
        self.POS_space = POS_space_length
        self.word_space = word_space_length
        self.lstm_dropout = 0.8

        self.initModel()

    def initModel(self):

        # POS Model
        self.model = Sequential()
        self.model.add(InputLayer((self.sentence_max,)))
        self.model.add(Embedding(self.word_space, 64))

        # pass hidden representation to LM
        aux_hidden_rep = self.model.add(Bidirectional(LSTM(256, dropout=self.lstm_dropout)))

        # LM Model
        self.model = Sequential()
        self.model.add(InputLayer((self.sentence_max,)))
        self.model.add(Embedding(self.word_space,64))

        # concatenate the aux. decoder's hidden representation with the first hidden layer
        h1 = self.model.add(Bidirectional(LSTM(256, dropout=self.lstm_dropout,
                                               return_sequence=True)))
        h1 = keras.layers.Concatenate(axis=1)([aux_hidden_rep, h1])

        self.model.add(Bidirectional(LSTM(128, return_sequences=True)))(h1)
        self.model.add(Bidirectional(LSTM(128)))
        self.model.add(TimeDistributed(Dense(self.word_space, activation='softmax')))

        self.model.compile(loss=self.loss, optimizer=self.optimizer, metrics=['accuracy'])
        self.model.summary()

    def loadCheckpoint(self, checkpoint):
        print('Loading Checkpoint: %s' % checkpoint)
        self.model = load_model(checkpoint)

    def saveCheckpoint(self, checkpoint):
        self.model.save(checkpoint)

    def train(self, trainX, trainY, testX, testY):
        self.model.fit(trainX, trainY, batch_size=self.batch_size, epochs=self.epochs,
                       validation_data=(testX, testY))

    def test(self, testX, testY):
        self.model.evaluate(testX, testY, batch_size=self.batch_size)