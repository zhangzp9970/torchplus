import torch
from typing import Optional, Union, Any, BinaryIO
import os
import pathlib

FILE_EXTENSIONS = ".dataz"


class DataZ(object):

    def __init__(
        self,
        data: torch.Tensor = None,
        label: Optional[Union[torch.Tensor, int]] = None,
    ) -> None:
        self.data = data
        self.label = label
        self.properties = dict()

    def set_data(self, data) -> None:
        self.data = data

    def get_data(self) -> torch.Tensor:
        return self.data

    def set_label(self, label: Optional[Union[torch.Tensor, int]]) -> None:
        self.label = label

    def get_label(self):
        return self.label

    def set_properties(self, prop_name: str, prop_value: Any) -> None:
        self.properties[prop_name] = prop_value

    def get_properties(self, prop_name: str) -> Any:
        return self.properties[prop_name]

    def save(self, fp: Union[str, pathlib.Path, BinaryIO] = None) -> None:
        dir = os.path.dirname(fp)
        os.makedirs(dir, exist_ok=True)
        file_extension = os.path.splitext(fp)[-1]
        if file_extension != FILE_EXTENSIONS:
            fp = fp + FILE_EXTENSIONS
        with open(fp, "wb") as f:
            torch.save(self, f)

    def load(self, fp: Union[str, pathlib.Path, BinaryIO] = None) -> None:
        dataz = torch.load(fp)
        self.data = dataz.data
        self.label = dataz.label
        self.properties = dataz.properties
