import datetime
import os
import random
from typing import Optional

import numpy as np
import torch
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter
from multiprocessing import cpu_count


class Init():
    def __init__(self, seed: int, log_root_dir: Optional[str] = None, backup_filename: Optional[str] = None, tensorboard: Optional[bool] = False) -> None:
        self.seed = seed
        self.log_root_dir = log_root_dir
        self.backup_filename = backup_filename
        self.tensorboard = tensorboard
        self.__init_seed(seed)
        if self.log_root_dir is not None:
            self.__set_log_dir(log_root_dir)
            if self.backup_filename is not None:
                self.__backup_file(self.backup_filename)
        if self.tensorboard is True:
            if os.path.exists(self.log_dir):
                self.writer = SummaryWriter(self.log_dir)
            else:
                self.writer = SummaryWriter()

    def __init_seed(self, seed: int) -> None:
        cudnn.benchmark = True
        cudnn.deterministic = True
        random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)

    def __set_log_dir(self, log_dir: str) -> None:
        now = datetime.datetime.now().strftime('%b%d_%H-%M-%S')
        self.log_dir = f'{log_dir}/{now}'
        os.makedirs(self.log_dir, exist_ok=True)

    def __backup_file(self, filename) -> None:
        with open(os.path.join(self.log_dir, os.path.basename(filename)), 'w', encoding='utf-8') as fout:
            with open(os.path.abspath(filename), 'r', encoding='utf-8') as fin:
                file = fin.read()
                fout.write(file)

    def get_device(self) -> torch.device:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def get_log_dir(self) -> str:
        try:
            return self.log_dir
        except AttributeError:
            raise RuntimeError('log directory not set!')

    def get_writer(self) -> SummaryWriter:
        try:
            return self.writer
        except AttributeError:
            raise RuntimeError('tensorboard summary writer not set!')

    def get_workers(self)->int:
        return cpu_count() if os.name == "posix" else 0
