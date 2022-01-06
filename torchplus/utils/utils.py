import torch
from torch.utils.data import Dataset, Subset
from typing import Optional


def class_split(dataset: Dataset, start: int, end: int, step: Optional[int] = 1) -> Subset:
    assert step > 0, 'step should be greater than 0'
    assert step < (end-start), 'length should be greater than step'
    ds_len = len(dataset)
    classes = torch.arange(start, end, step)
    indices = list(range(ds_len))
    selected_indices = []
    for i in indices:
        if dataset.targets[i] in classes:
            selected_indices.append(indices[i])
    return Subset(dataset, selected_indices)
