class History:
    def __init__(self):
        self.history = []

    def register(self, data):
        self.history.append(data)