from sklearn.dummy import DummyClassifier

class TrivialModel:

    model = None

    def __init__(self):
        self.initModel()

    def getModel(self):
        model = DummyClassifier(strategy="most_frequent")
        return model
        

    def initModel(self):
        self.model = self.getModel()
        return self.model

    def train(self, trainX, trainY):
        self.model.fit(trainX, trainY)
    
    def test(self, testX, testY):
        self.model.score(testX, testY)
        
    def predict(self, X):
        return self.model.predict(X)
