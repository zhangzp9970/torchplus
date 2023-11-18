import os
from typing import Any, Callable, Optional
from PIL import Image
import torch
from torch.utils.data import DataLoader, TensorDataset
from torchvision.datasets import VisionDataset, ImageFolder
from tqdm import tqdm


class FlatFolder(VisionDataset):
    def __init__(
        self,
        root: str,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
    ) -> None:
        super().__init__(root, transform=transform, target_transform=target_transform)
        self.filename = os.listdir(root)

    def __getitem__(self, index: int) -> Any:
        X = Image.open(os.path.join(self.root, self.filename[index])).convert("RGB")
        if self.transform is not None:
            X = self.transform(X)
        return X

    def __len__(self) -> int:
        return len(self.filename)


def PreProcessFolder(
    root: str,
    transform: Optional[Callable] = None,
    target_transform: Optional[Callable] = None,
    loader: Callable[[str], Any] = ImageFolder,
    batch_size=128,
    num_workers=2,
):
    ds = loader(root=root, transform=transform, target_transform=target_transform)
    train_dl = DataLoader(
        dataset=ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        drop_last=False,
    )
    if loader == ImageFolder:
        imlist = []
        labellist = []
        for i, (im, label) in enumerate(tqdm(train_dl, desc=f"pre-process dataset")):
            imlist.append(im)
            labellist.append(label)
        imlist = torch.cat(imlist)
        labellist = torch.cat(labellist)
        ds = TensorDataset(imlist, labellist)
    elif loader == FlatFolder:
        imlist = []
        for i, im in enumerate(tqdm(train_dl, desc=f"pre-process dataset")):
            imlist.append(im)
        imlist = torch.cat(imlist)
        ds = TensorDataset(imlist)
    else:
        raise ValueError("loader not found! Use ImageFolder or FlatFolder.")

    return ds
