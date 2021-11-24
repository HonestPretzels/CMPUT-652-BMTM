# TODO: identify the vocab length, add it into the consts script, then here
from consts import word_space_length, sentence_max_length # , vocab_length
from keras.models import Sequential, load_model
from keras.layers import Dense, LSTM, InputLayer, Bidirectional,  TimeDistributed, Embedding, Activation
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.grid_search import GridSearchCV

class LM_Model:

    model = None

    def __init__(self):
        self.optimizer = "Adam"
        self.loss = 'sparse_categorical_crossentropy'
        self.learning_rate = np.array([0.001, 0.01, 0.05, 0.1])
        self.batch_size = np.array([20, 40, 60, 80]) # Lan et al. used 20 for PTB data set
        self.epochs = np.array([50, 100, 250, 500]) # Lan et al. used 500 epochs for PTB data set
        self.validation_split = 0.2
        self.sentence_max = sentence_max_length
        self.word_space = word_space_length
        self.lstm_dropout = 0.8

        self.initModel()

    def initModel(self):
        model = None
        self.model = Sequential()
        self.model.add(InputLayer((self.sentence_max,)))
        self.model.add(Embedding(self.word_space, 64))  # Is 64 only for PTB?

        self.model.add(Bidirectional(LSTM(128, return_sequences=True)))
        self.model.add(Bidirectional(LSTM(128, return_sequences=True)))
        self.model.add(Bidirectional(LSTM(128)))

        # TODO: ensure vocab_length is imported into this file
        self.model.add(Dense(self.word_space, activation='softmax'))

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

    # only added gridsearch for this model - if works well, then we can use it for poslm             
    def gridSearch(self):
        start = time()
        model = KerasClassifier(build_fn = self.initModel)
        param_grid = dict(self.learning_rate, nb_epoch = self.epochs, batch_size = self.batches)
        grid = GridSearchCV(estimator=model, param_grid=param_grid)
        grid_result = grid.fit(trainX, trainY)
        print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
        for params, mean_score, scores in grid_result.grid_scores_:
            print("LM mean: %f and std: (%f) with: %r" % (scores.mean(), scores.std(), params))
        print("total time for LM:", time()-start)
                
    def loadCheckpoint(self, checkpoint):
        print('Loading Checkpoint: %s'%checkpoint)
        self.model = load_model(checkpoint)

    def saveCheckpoint(self, checkpoint):
        self.model.save(checkpoint)

    def train(self, trainX, trainY, testX, testY):
        self.model.fit(trainX, trainY, batch_size=self.batch_size, epochs=self.epochs,
                       validation_data=(testX, testY))
    
    def test(self, testX, testY):
        self.model.evaluate(testX, testY, batch_size=self.batch_size)
        # TODO: Fix metrics to show more than just accuracy