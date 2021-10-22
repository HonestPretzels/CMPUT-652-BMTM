class LM_Model:

    model = None

    def __init__(self):
        self.optimizer = "Adam"

    def initModel(self):
        model = None
        # TODO: Keras stuff to create the model