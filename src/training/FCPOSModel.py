# TODO: identify the vocab length, add it into the consts script, then here
from consts import POS_space_length, sentence_max_length # , vocab_length
from keras.models import Sequential
import tensorflow as tf
from keras.layers import Dense, InputLayer, Embedding

class FC_POS_Model:

    model = None

    def __init__(self):
        self.optimizer = "Adam"
        self.loss = 'categorical_crossentropy'
        # self.learning_rate = np.array([0.001, 0.01, 0.05, 0.1])
        # self.batch_size = np.array([20, 40, 60, 80]) # Lan et al. used 20 for PTB data set
        # self.epochs = np.array([50, 100, 250, 500]) # Lan et al. used 500 epochs for PTB data set
        # self.hidden_nodes = np.array([128, 300])
        self.batch_size = 50
        self.learning_rate = 0.001
        self.batch_size = 50
        self.epochs = 90
        self.validation_split = 0.2
        self.sentence_max = sentence_max_length
        self.word_space = POS_space_length
        self.lstm_dropout = 0.8

        self.initModel()

    def getModel(self):
        model = Sequential()
        model.add(InputLayer((self.sentence_max,)))
        model.add(Embedding(self.word_space, 64))  # Is 64 only for PTB?
        model.add(Dense(256, activation="relu"))

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
