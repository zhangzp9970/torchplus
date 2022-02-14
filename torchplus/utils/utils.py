import torch
from torch.utils.data import Dataset, Subset
from typing import Optional
import pandas as pd
from PIL import Image
from torchvision.transforms.functional import to_tensor, to_grayscale


def class_split(dataset: Dataset, start: int, end: int, step: Optional[int] = 1) -> Subset:
    assert step > 0, 'step should be greater than 0'
    assert step <= (end-start), 'length should be greater than step'
    ds_len = len(dataset)
    classes = torch.arange(start, end, step)
    indices = list(range(ds_len))
    selected_indices = []
    for i in indices:
        if dataset.targets[i] in classes:
            selected_indices.append(indices[i])
    return Subset(dataset, selected_indices)


def save_excel(tensor: torch.Tensor, path: str) -> None:
    with pd.ExcelWriter(path) as Ewriter:
        if tensor.dim() == 3:
            for i in range(tensor.shape[0]):
                t = tensor[i]
                data = t.detach().cpu().numpy()
                df = pd.DataFrame(data)
                df.to_excel(Ewriter, sheet_name=str(i),
                            index=False, header=False)
        elif tensor.dim() >= 4:
            raise RuntimeError("tensor shape should be less than 3")
        else:
            if tensor.dim() == 0:
                tensor = tensor.reshape(1)
            data = tensor.detach().cpu().numpy()
            df = pd.DataFrame(data)
            df.to_excel(Ewriter, sheet_name=str(0), index=False, header=False)


def read_image_to_tensor(path: str, grayscale: bool = False) -> torch.Tensor:
    img = Image.open(path)
    if grayscale:
        img = to_grayscale(img, num_output_channels=1)
    im = to_tensor(img)
    return im
