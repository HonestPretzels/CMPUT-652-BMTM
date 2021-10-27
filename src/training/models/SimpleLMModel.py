from keras.models import load_model

class LM_Model:

    model = None

    def __init__(self):
        # TODO: Initialize the hyper-parameters
        self.optimizer = 'Adam'
        self.loss = 'categorical_crossentropy'
        self.learning_rate = 0.001
        self.batch_size = 128
        self.epochs = 50
        self.validation_split = 0.2

        self.initModel()

    def initModel(self):
        # TODO: Implement the model here
        pass

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
