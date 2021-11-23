from consts import word_space_length, POS_space_length, sentence_max_length
from keras.models import Model, Sequential, load_model
from keras.layers import Dense, LSTM, InputLayer, Bidirectional, \
    TimeDistributed, Embedding, Activation, Concatenate, Input


# Help taken from here https://www.pyimagesearch.com/2018/06/04/keras-multiple-outputs-and-multiple-losses/

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
        input_layer = Input((self.sentence_max,))
        aux_model = self.posHiddenRep(input_layer)
        lm_model = self.lmHiddenRep(input_layer,None)
        print(aux_model)
        print(lm_model)
        self.model = Model(inputs=input_layer, outputs=[aux_model,lm_model], name="POSLM_Model")
        losses = {
            "posModel": "categorical_crossentropy",
            "lmModel": "sparse_categorical_crossentropy"
        }
        lossWeights = {"posModel": 1.0, "lmModel": 1.0}
        self.model.compile(optimizer=self.optimizer, loss=losses, loss_weights=lossWeights, metrics=["accuracy"])
        self.model.summary()

    def posHiddenRep(self, inputLayer):
        # This is the POS model on it's own, we
        aux_model = Embedding(self.word_space, 64)(inputLayer)
        aux_model = Bidirectional(LSTM(256, dropout=self.lstm_dropout, return_sequences=True))(aux_model)
        aux_model = TimeDistributed(Dense(self.POS_space))(aux_model)
        aux_model = Activation('softmax', name="posModel")(aux_model)

        return aux_model


    def lmHiddenRep(self, inputLayer, aux_layer):
        lm_model = Embedding(self.word_space,64)(inputLayer)
        lm_model = Bidirectional(LSTM(128, return_sequences=True))(lm_model)
        lm_model = Bidirectional(LSTM(128, return_sequences=True))(lm_model)
        lm_model = Bidirectional(LSTM(128))(lm_model)
        lm_model = Dense(self.word_space)(lm_model)
        lm_model = Activation('softmax', name="lmModel")(lm_model)
        

        # concatenate the aux. decoder's hidden representation with the first hidden layer
        # h1 = self.model.add(Bidirectional(LSTM(256, dropout=self.lstm_dropout, return_sequences=True)))
        # h1 = Concatenate()([auxiliary_hidden_representation, h1])

        # self.model.add(Bidirectional(LSTM(128, return_sequences=True)))(h1)
        # self.model.add(Bidirectional(LSTM(128, return_sequences=True)))
        # self.model.add(TimeDistributed(Dense(self.word_space, activation='softmax')))

        # self.model.compile(loss=self.loss, optimizer=self.optimizer, metrics=['accuracy'])
        # self.model.summary()
        return lm_model


    def loadCheckpoint(self, checkpoint):
        print('Loading Checkpoint: %s' % checkpoint)
        self.model = load_model(checkpoint)

    def saveCheckpoint(self, checkpoint):
        self.model.save(checkpoint)

    def train(self, trainX, trainYPOS, trainYLM, testX, testYPOS, testYLM):
        self.model.fit(x=trainX, 
                       y={"posModel":trainYPOS, "lmModel":trainYLM}, 
                       batch_size=self.batch_size, epochs=self.epochs, verbose=1,
                       validation_data=(testX, {"posModel":testYPOS, "lmModel":testYLM}))

    # def test(self, testX, testY):
    #     self.model.evaluate(testX, testY, batch_size=self.batch_size)