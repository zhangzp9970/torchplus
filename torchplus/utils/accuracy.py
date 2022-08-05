from typing import Optional, Dict, TypeVar
import torch
import torch.nn.functional as F

T_torchplus=TypeVar('T_torchplus',torch.Tensor,Dict)

class BaseAccuracy(object):
    def __init__(self, class_number: int) -> None:
        self.count_pool = torch.zeros(class_number)
        self.value_pool = torch.zeros(class_number)
        self.accuracy_pool = torch.zeros(class_number)

    def accumulate(self, label: torch.Tensor, value: torch.Tensor) -> None:
        assert label.shape == value.shape
        assert label.device==value.device
        output_device = label.device
        self.count_pool = self.count_pool.to(label.device)
        self.value_pool = self.value_pool.to(output_device)
        self.accuracy_pool = self.accuracy_pool.to(output_device)
        for i in range(len(label)):
            self.count_pool[label[i]] += torch.tensor(1.0).to(output_device)
            self.value_pool[label[i]] += value[i]

        for i in range(len(self.count_pool)):
            self.accuracy_pool[i] = self.value_pool[i]/self.count_pool[i]

    def get(self, per_class: Optional[bool] = False, isdict: Optional[bool] = False) -> T_torchplus:
        if per_class:
            if isdict:
                accuracy_dict = dict()
                for i in range(len(self.accuracy_pool)):
                    accuracy_dict[str(i)] = self.accuracy_pool[i]
                return accuracy_dict
            else:
                return self.accuracy_pool
        else:
            return torch.sum(self.value_pool)/torch.sum(self.count_pool)


class ClassificationAccuracy(BaseAccuracy):
    def __init__(self, class_number: int) -> None:
        super().__init__(class_number)

    def accumulate(self, label: torch.Tensor, predict: torch.Tensor) -> None:
        assert label.shape == predict.shape
        value = label == predict
        value=value.to(torch.float)
        super().accumulate(label, value)
