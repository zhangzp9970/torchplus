import torch

class ClassificationAccuracy():
    def __init__(self) -> None:
        self.correct=0.0
        self.total=0.0

    def compute_accuracy_minibatch():
        raise NotImplementedError()

    def compute_accuracy_total():
        raise NotImplementedError()