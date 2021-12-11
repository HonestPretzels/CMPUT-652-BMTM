from consts import word_space_length, POS_space_length, sentence_max_length
import tensorflow as tf
from keras.models import Model
from keras.layers import Dense, LSTM, Bidirectional, \
    TimeDistributed, Embedding, Activation, Concatenate, Input


# Help taken from here https://www.pyimagesearch.com/2018/06/04/keras-multiple-outputs-and-multiple-losses/

class POSLM_Model:

    model = None

    def __init__(self):
        self.learning_rate = 0.001
        self.loss = 'categorical_crossentropy'
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
        self.batch_size = 50
        self.epochs = 90
        self.validation_split = 0.2
        self.sentence_max = sentence_max_length
        self.POS_space = POS_space_length
        self.word_space = word_space_length
        self.lstm_dropout = 0.8

        self.initModel()

    def initModel(self):
        input_layer = Input((self.sentence_max,))
        aux_model, aux_layer = self.posHiddenRep(input_layer)
        lm_model, _ = self.lmHiddenRep(input_layer,aux_layer)
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
        aux_layer = Bidirectional(LSTM(64, dropout=self.lstm_dropout, return_sequences=True))(aux_model)
        aux_model = TimeDistributed(Dense(self.POS_space))(aux_layer)
        aux_model = Activation('softmax', name="posModel")(aux_model)

        return aux_model, aux_layer


    def lmHiddenRep(self, inputLayer, aux_layer):
        lm_model = Embedding(self.word_space,128)(inputLayer)
        lm_model = Concatenate(axis=2)([lm_model, aux_layer])
        lm_model = Bidirectional(LSTM(128, dropout=self.lstm_dropout, return_sequences=True))(lm_model)
        lm_model = Bidirectional(LSTM(128, dropout=self.lstm_dropout, return_sequences=True))(lm_model)
        hiddenRep = Bidirectional(LSTM(128, dropout=self.lstm_dropout))(lm_model)
        lm_model = Dense(self.word_space)(hiddenRep)
        lm_model = Activation('softmax', name="lmModel")(lm_model)
       
        return lm_model, hiddenRep


    def loadCheckpoint(self, checkpoint):
        print('Loading Checkpoint: %s' % checkpoint)
        self.model.load_weights(checkpoint)

    def saveCheckpoint(self, checkpoint):
        self.model.save(checkpoint)

    def loadHyperParameters(self, config_dict):
        for key in config_dict:
            try:
                getattr(self, key)
                setattr(self, key, config_dict[key])
                if key == "learning_rate":
                    self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
            except:
                continue

    def train(self, trainX, trainYPOS, trainYLM, testX, testYPOS, testYLM, checkpointPath):
        model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpointPath,
            save_weights_only=True,
            monitor='val_lmModel_accuracy',
            mode='max',
            save_best_only=True)
        
        self.model.fit(x=trainX, 
                        y={"posModel":trainYPOS, "lmModel":trainYLM}, 
                        batch_size=self.batch_size, epochs=self.epochs, verbose=1,
                        validation_data=(testX, {"posModel":testYPOS, "lmModel":testYLM}),
                        callbacks=[model_checkpoint_callback])

    def test(self, testX, testY):
        self.model.evaluate(testX, testY, batch_size=self.batch_size)
    
    def predict(self, X):
        return self.model.predict(X, batch_size=self.batch_size)