import os
from typing import Any, Callable, Optional
from PIL import Image
from torchvision.datasets import VisionDataset


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
