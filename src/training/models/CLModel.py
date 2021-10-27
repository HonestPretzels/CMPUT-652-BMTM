class CL_Model:

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
        # TODO: Implement SPACENET here
        pass

    def loadHyperParameters(self, config_dict):
        # TODO: Load the hyper parameters
        pass

    def loadCheckpoint(self, checkpoint):
        # TODO: Load a checkpoint into the model
        pass

    def saveCheckpoint(self, checkpoint):
        # TODO: Save a checkpoint
        pass

    def train(self, trainX, trainY):
        self.model.fit(trainX, trainY, batch_size=self.batch_size, epochs=self.epochs, validation_split=self.validation_split)
    
    def test(self, testX, testY):
        pred = self.model.predict(testX)

        # TODO: Implement metrics
