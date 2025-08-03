import torch
import torch.nn as nn
import os
import pathlib
from torch.utils.data import Dataset, Subset
from typing import Any, BinaryIO, List, Optional, Union
import pandas as pd
from torchvision.utils import save_image
from base64 import b64encode


def class_split(
    dataset: Dataset, start: int, end: int, step: Optional[int] = 1
) -> Subset:
    assert step > 0, "step should be greater than 0"
    assert step <= (end - start), "length should be greater than step"
    ds_len = len(dataset)
    classes = torch.arange(start, end, step)
    indices = list(range(ds_len))
    selected_indices = []
    for i in indices:
        if dataset[i][1] in classes:
            selected_indices.append(indices[i])
    return Subset(dataset, selected_indices)


def save_excel(tensor: torch.Tensor, path: str) -> None:
    dir = os.path.dirname(path)
    os.makedirs(dir, exist_ok=True)
    with pd.ExcelWriter(path) as Ewriter:
        if tensor.dim() == 3:
            for i in range(tensor.shape[0]):
                t = tensor[i]
                data = t.detach().cpu().numpy()
                df = pd.DataFrame(data)
                df.to_excel(Ewriter, sheet_name=str(i), index=False, header=False)
        elif tensor.dim() >= 4:
            raise RuntimeError("tensor shape should be less than 3")
        else:
            if tensor.dim() == 0:
                tensor = tensor.reshape(1)
            data = tensor.detach().cpu().numpy()
            df = pd.DataFrame(data)
            df.to_excel(Ewriter, sheet_name=str(0), index=False, header=False)


def save_csv(tensor: torch.Tensor, path: str, sep: Optional[str] = ",") -> None:
    dir = os.path.dirname(path)
    os.makedirs(dir, exist_ok=True)
    if tensor.dim() >= 3:
        raise RuntimeError("tensor shape should be less than 2")
    else:
        if tensor.dim() == 0:
            tensor = tensor.reshape(1)
        data = tensor.detach().cpu().numpy()
        df = pd.DataFrame(data)
        df.to_csv(path, sep=sep, index=False, header=False)


def save_image2(
    tensor: Union[torch.Tensor, List[torch.Tensor]],
    fp: Union[str, pathlib.Path, BinaryIO],
    format: Optional[str] = None,
    **kwargs,
) -> None:
    dir = os.path.dirname(fp)
    os.makedirs(dir, exist_ok=True)
    save_image(tensor, fp, format, **kwargs)

def hash_code(obj: object) -> str:
    return b64encode(str(hash(obj)).encode()).decode()[0:7]


def __guassian_kernel(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    n_samples = int(source.size()[0]) + int(target.size()[0])
    total = torch.cat([source, target], dim=0)
    total0 = total.unsqueeze(0).expand(
        int(total.size(0)), int(total.size(0)), int(total.size(1))
    )
    total1 = total.unsqueeze(1).expand(
        int(total.size(0)), int(total.size(0)), int(total.size(1))
    )
    L2_distance = ((total0 - total1).pow(2)).sum(2)
    if fix_sigma:
        bandwidth = fix_sigma
    else:
        bandwidth = torch.sum(L2_distance.data) / (n_samples**2 - n_samples)
    bandwidth /= kernel_mul ** (kernel_num // 2)
    bandwidth_list = [bandwidth * (kernel_mul**i) for i in range(kernel_num)]
    kernel_val = [
        torch.exp(-L2_distance / bandwidth_temp) for bandwidth_temp in bandwidth_list
    ]
    return sum(kernel_val)


def MMD(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None)->torch.Tensor:
    batch_size = int(source.size()[0])
    kernels = __guassian_kernel(
        source,
        target,
        kernel_mul=kernel_mul,
        kernel_num=kernel_num,
        fix_sigma=fix_sigma,
    )
    XX = kernels[:batch_size, :batch_size]
    YY = kernels[batch_size:, batch_size:]
    XY = kernels[:batch_size, batch_size:]
    YX = kernels[batch_size:, :batch_size]
    loss = torch.mean(XX + YY - XY - YX)
    return loss


def model_size(model: nn.Module):
    # calculate model size. Reference: https://discuss.pytorch.org/t/finding-model-size/130275/2
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    size_all_mb = (param_size + buffer_size) / 1024**2
    return size_all_mb
