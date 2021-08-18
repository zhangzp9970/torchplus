import os
from typing import Any, Callable, Optional, Tuple

from PIL import Image
from torchvision.datasets.vision import VisionDataset
from torchvision.transforms.transforms import Compose, Grayscale, Resize, ToTensor
from torchvision.datasets.utils import check_integrity, download_and_extract_archive, extract_archive


class KSDD(VisionDataset):
    base_folder = "KolektorSDD"
    url = "http://go.vicos.si/kolektorsdd"
    file_name = "KolektorSDD.zip"
    zip_md5 = "2b094030343c1cd59df02203ac6c57a0"
    train_dir_list = []
    test_dir_list = []
    data_list = []
    X = []
    Y = []
    PoN = []
    Xpos = []
    Ypos = []
    PoNpos = []
    GetX = []
    GetY = []
    GetPoN = []

    basic_transform = Compose(
        [
            Grayscale(num_output_channels=1),
            Resize((1408, 512))
        ]
    )

    def __init__(self, root: str, train: bool = True, fold: Optional[int] = 3, transform: Optional[Callable] = None, target_transform: Optional[Callable] = None, download: bool = False, positive_only: Optional[bool] = False) -> None:
        super(KSDD, self).__init__(root=root, transform=transform,
                                   target_transform=target_transform)
        self.root = root
        self.train = train
        self.fold = fold
        self.transform = transform
        self.target_transform = target_transform
        self.download = download
        self.positive_only = positive_only
        if self.download:
            self.__download()

        if not self.__check_integrity():
            raise RuntimeError('Dataset not found or corrupted.' +
                               ' You can use download=True to download it')
        if not os.path.exists(os.path.join(self.root, self.base_folder)):
            print('Target directory not found')
            print("Extracting {} to {}".format(
                os.path.join(self.root, self.file_name), os.path.join(self.root, self.base_folder)))
            extract_archive(os.path.join(self.root, self.file_name),
                            os.path.join(self.root, self.base_folder))
        self.__make_dir_lists(root=os.path.join(
            self.root, self.base_folder), fold=self.fold)
        if self.train:
            dir_list = self.train_dir_list
        else:
            dir_list = self.test_dir_list
        self.__make_item_lists(dir_list)
        self.__make_pon_lists()
        if self.positive_only:
            self.__make_p_lists()
            self.GetX = self.Xpos
            self.GetY = self.Ypos
            self.GetPoN = self.PoNpos
        else:
            self.GetX = self.X
            self.GetY = self.Y
            self.GetPoN = self.PoN

    def __make_dir_lists(self, root: str, fold: int = 3) -> None:
        total_dir_list = os.listdir(root)
        total_dir_list_len = len(total_dir_list)
        train_dir_list_len = int(total_dir_list_len*(1.0-1.0/float(fold)))
        test_dir_list_len = total_dir_list_len-train_dir_list_len
        for i in range(train_dir_list_len):
            self.train_dir_list.append(total_dir_list[i])
        for i in range(test_dir_list_len):
            i = i+train_dir_list_len
            self.test_dir_list.append(total_dir_list[i])

    def __make_item_lists(self, dir_list: list) -> None:
        for folder_name in dir_list:
            file_list = os.listdir(os.path.join(
                self.root, self.base_folder, folder_name))
            for i, file in enumerate(file_list):
                self.X.append(os.path.join(self.root, self.base_folder, folder_name, file)) if i % 2 == 0 else self.Y.append(
                    os.path.join(self.root, self.base_folder, folder_name, file))

    def __make_pon_lists(self) -> None:
        for y in self.Y:
            self.PoN.append(True if self.__is_positive(y) else False)

    def __make_p_lists(self) -> None:
        for i, pon in enumerate(self.PoN):
            if pon:
                self.Xpos.append(self.X[i])
                self.Ypos.append(self.Y[i])
                self.PoNpos.append(self.PoN[i])

    def __getitem__(self, index: int) -> Tuple[Any, Any, Any]:
        img = Image.open(self.GetX[index])
        label = Image.open(self.GetY[index])
        pon = self.GetPoN[index]
        img = self.basic_transform(img)
        label = self.basic_transform(label)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            label = self.target_transform(label)

        return img, label, pon

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

    def __is_positive(self, y) -> bool:
        label = Image.open(y)
        label = self.basic_transform(label)
        label = ToTensor()(label)
        if True in (label > 0):
            return True
        else:
            return False
