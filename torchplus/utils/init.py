import datetime
import os
import random

import numpy as np
import torch
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter


def init_seed(seed: int) -> None:
    cudnn.benchmark = True
    cudnn.deterministic = True
    seed = 9970
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


def auto_save_file(filename, target_dir) -> None:
    with open(os.path.join(target_dir, os.path.basename(filename)), 'w', encoding='utf-8') as fout:
        with open(os.path.abspath(filename), 'r', encoding='utf-8') as fin:
            file = fin.read()
            fout.write(file)


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_summary_writer(root_dir: str = None) -> SummaryWriter:
    if root_dir is not None:
        now = datetime.datetime.now().strftime('%b%d_%H-%M-%S')
        log_dir = f'{root_dir}/{now}'
        return SummaryWriter(log_dir)
    else:
        return SummaryWriter()
