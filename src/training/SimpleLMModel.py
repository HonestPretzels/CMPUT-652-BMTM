from consts import word_space_length, sentence_max_length
import tensorflow as tf
from keras.models import Sequential, load_model
from keras.layers import Dense, LSTM, InputLayer, Bidirectional, Embedding

class LM_Model:

    model = None

    def __init__(self):
        self.optimizer = "Adam"
        self.loss = 'sparse_categorical_crossentropy'
        self.batch_size = 50
        self.learning_rate = 0.001
        self.batch_size = 50
        self.epochs = 90
        self.validation_split = 0.2
        self.sentence_max = sentence_max_length
        self.word_space = word_space_length
        self.lstm_dropout = 0.8

        self.initModel()

    def getModel(self, n_hidden=128):
        model = Sequential()
        model.add(InputLayer((self.sentence_max,)))
        model.add(Embedding(self.word_space, 64))  # Is 64 only for PTB?

        model.add(Bidirectional(LSTM(n_hidden, return_sequences=True)))
        model.add(Bidirectional(LSTM(n_hidden, return_sequences=True)))
        model.add(Bidirectional(LSTM(n_hidden)))

        # TODO: ensure vocab_length is imported into this file
        model.add(Dense(self.word_space, activation='softmax'))
        return model
        

    def initModel(self):
        self.model = self.getModel()

        self.model.compile(loss=self.loss, optimizer=self.optimizer, metrics=['accuracy'])
        self.model.summary()
        return self.model

    def loadHyperParameters(self, config_dict):
        for key in config_dict:
            try:
                getattr(self, key)
                setattr(self, key, config_dict[key])
                if key == "learning_rate":
                    self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
            except:
                continue
                
    def loadCheckpoint(self, checkpoint):
        print('Loading Checkpoint: %s'%checkpoint)
        self.model.load_weights(checkpoint)


    def saveCheckpoint(self, checkpoint):
        self.model.save(checkpoint)

    def train(self, trainX, trainY, testX, testY, checkpointPath):
        model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpointPath,
            save_weights_only=True,
            monitor='val_accuracy',
            mode='max',
            save_best_only=True)
        
        self.model.fit(trainX, trainY, batch_size=self.batch_size,
                       epochs=self.epochs, validation_data=(testX, testY),
                       callbacks=[model_checkpoint_callback])
    
    def test(self, testX, testY):
        self.model.evaluate(testX, testY, batch_size=self.batch_size)
        
    def predict(self, X):
        return self.model.predict(X, batch_size=self.batch_size)