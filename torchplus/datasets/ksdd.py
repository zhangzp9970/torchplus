import os
from typing import Any, Callable, Optional, Tuple

from PIL import Image
from torchvision.datasets.vision import VisionDataset
from torchvision.transforms.transforms import Compose, Grayscale, Resize
from torchvision.datasets.utils import check_integrity, download_and_extract_archive,extract_archive


class KSDD(VisionDataset):
    base_folder = "KolektorSDD"
    url = "http://go.vicos.si/kolektorsdd"
    file_name = "KolektorSDD.zip"
    zip_md5 = "2b094030343c1cd59df02203ac6c57a0"
    train_list = []
    test_list = []
    data_list = []
    X = []
    Y = []
    basic_transform = Compose(
        [
            Grayscale(num_output_channels=1),
            Resize((1408, 512))
        ]
    )

    def __init__(self, root: str, train: bool = True, fold: Optional[int] = 3, transform: Optional[Callable] = None, target_transform: Optional[Callable] = None, download: bool = False) -> None:
        super(KSDD, self).__init__(root=root, transform=transform,
                                   target_transform=target_transform)
        self.train = train
        if download:
            self.__download()

        if not self.__check_integrity():
            raise RuntimeError('Dataset not found or corrupted.' +
                               ' You can use download=True to download it')
        if not os.path.exists(os.path.join(root, self.base_folder)):
            print('Target directory not found')
            print("Extracting {} to {}".format(
                os.path.join(root, self.file_name), os.path.join(self.root, self.base_folder)))
            extract_archive(os.path.join(root,self.file_name), os.path.join(self.root, self.base_folder))
        self.__make_lists(root=os.path.join(
            root, self.base_folder), fold=fold)
        if self.train:
            data_list = self.train_list
        else:
            data_list = self.test_list

        for folder_name in data_list:
            file_list = os.listdir(os.path.join(
                root, self.base_folder, folder_name))
            for i, file in enumerate(file_list):
                self.X.append(os.path.join(root, self.base_folder, folder_name, file)) if i % 2 == 0 else self.Y.append(
                    os.path.join(root, self.base_folder, folder_name, file))

    def __make_lists(self, root: str, fold: int = 3) -> None:
        total_list = os.listdir(root)
        total_list_len = len(total_list)
        train_list_len = int(total_list_len*(1.0-1.0/float(fold)))
        test_list_len = total_list_len-train_list_len
        for i in range(train_list_len):
            self.train_list.append(total_list[i])
        for i in range(test_list_len):
            i = i+train_list_len
            self.test_list.append(total_list[i])

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        img = Image.open(self.X[index])
        label = Image.open(self.Y[index])
        img = self.basic_transform(img)
        label = self.basic_transform(label)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            label = self.target_transform(label)

        return img, label

    def __len__(self) -> int:
        return len(self.X)

    def __download(self) -> None:
        if self.__check_integrity():
            print('Files already downloaded and verified')
            return
        download_and_extract_archive(
            self.url, self.root, extract_root=os.path.join(self.root, self.base_folder), filename=self.file_name, md5=self.zip_md5)

    def __check_integrity(self) -> bool:
        file_path = os.path.join(self.root, self.file_name)
        if not check_integrity(file_path, self.zip_md5):
            return False
        return True
