# TODO: identify the vocab length, add it into the consts script, then here
import keras
from consts import word_space_length, sentence_max_length # , vocab_length
import numpy as np
import time
from keras.models import Sequential, load_model
from keras.layers import Dense, LSTM, InputLayer, Bidirectional,  TimeDistributed, Embedding, Activation
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV

class LM_Model:

    model = None

    def __init__(self):
        self.optimizer = "Adam"
        self.loss = 'sparse_categorical_crossentropy'
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
            except:
                continue

    # only added gridsearch for this model - if works well, then we can use it for poslm             
    def gridSearchCV(self, trainX, trainY):
        start = time.time()
        model = KerasClassifier(build_fn = self.initModel)
        param_grid = dict(epochs=[5], batch_size = self.batch_size, n_hidden = self.hidden_nodes)
        print(param_grid)
        grid = GridSearchCV(estimator=model, param_grid=param_grid, scoring='accuracy') # default is 3-fold CV
        grid_result = grid.fit(trainX, trainY)
        print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
        for params, mean_score, scores in grid_result.grid_scores_:
            print("LM mean: %f and std: (%f) with: %r" % (scores.mean(), scores.std(), params))
        print("total time for LM:", time.time()-start)
        return grid
                
    def loadCheckpoint(self, checkpoint):
        print('Loading Checkpoint: %s'%checkpoint)
        self.model.load_weights(checkpoint)


    def saveCheckpoint(self, checkpoint):
        self.model.save(checkpoint)

    def train(self, trainX, trainY, testX, testY):
        self.model.fit(trainX, trainY, batch_size=self.batch_size, epochs=self.epochs,
                       validation_data=(testX, testY))
        # self.gridSearchCV(trainX,trainY)
        
        # now that we have gridsearch, we need to use the best parameters to for fitting
        #grid = grid.fit(trainX, trainY)
    
    def test(self, testX, testY):
        self.model.evaluate(testX, testY, batch_size=self.batch_size)
        # TODO: Fix metrics to show more than just accuracy
        
    def predict(self, X):
        return self.model.predict(X, batch_size=self.batch_size)

    def goToHiddenRep(self):
        m = self.getModel()
        extractor = keras.Model(inputs = m.inputs, outputs=[m.layers[3].output])
        self.model = extractor
        self.model.compile(optimizer=self.optimizer, metrics=["accuracy"])
        self.model.summary()
