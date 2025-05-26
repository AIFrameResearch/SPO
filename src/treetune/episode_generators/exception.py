class NoTrainingDataException(Exception):
    def __init__(self, message="No training samples available."):
        self.message = message
        super().__init__(self.message)