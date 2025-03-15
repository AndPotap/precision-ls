class Task:
    def __init__(self, name=None, **kwargs):
        self.name = name

    # For use in training, includes task formatting. Output a dictionary of tensors: prefix, x, and y.
    def evaluate(self, x):
        raise NotImplementedError

    # Can be used separately from training, without formatting.
    def out(self, x):
        raise NotImplementedError

    @staticmethod
    def get_metric():
        raise NotImplementedError

    @staticmethod
    def get_training_metric():
        raise NotImplementedError
