from typing import Optional
import torch


class ClassificationAccuracy():
    def __init__(self, class_number: int) -> None:
        self.correct_pool = torch.tensor([0.0 for x in range(class_number)])
        self.total_pool = torch.tensor([0.0 for x in range(class_number)])
        self.accuracy_pool = torch.tensor([0.0 for x in range(class_number)])

    def accumulate(self, label: torch.Tensor, predict: torch.Tensor) -> None:
        assert label.shape == predict.shape
        for i in range(len(label)):
            self.total_pool[label[i]] += torch.tensor(1.0)
            if label[i] == predict[i]:
                self.correct_pool[label[i]] += torch.tensor(1.0)

        for i in range(len(self.total_pool)):
            self.accuracy_pool[i] = self.correct_pool[i]/self.total_pool[i]

    def get(self, per_class: Optional[bool] = False) -> torch.Tensor:
        if per_class:
            return self.accuracy_pool
        else:
            return torch.sum(self.correct_pool)/torch.sum(self.total_pool)
