import datetime
import os
import random
from typing import Optional
import warnings

import numpy as np
import torch
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter
from multiprocessing import cpu_count
from torch.profiler import *


class Init():
    def __init__(self, seed: int, log_root_dir: Optional[str] = None, backup_filename: Optional[str] = None, tensorboard: Optional[bool] = False, profiler: Optional[bool] = False, **kwargs) -> None:
        self.seed = seed
        self.log_root_dir = log_root_dir
        self.backup_filename = backup_filename
        self.tensorboard = tensorboard
        self.profiler = profiler
        self.__init_seed(seed)
        if self.log_root_dir is not None:
            self.__set_log_dir(log_root_dir)
            if self.backup_filename is not None:
                self.__backup_file(self.backup_filename)
        else:
            if self.backup_filename is not None:
                warnings.warn('Log directory is NONE, file not backuped!')
        if self.tensorboard is True:
            if os.path.exists(self.log_dir):
                self.writer = SummaryWriter(self.log_dir)
            else:
                self.writer = SummaryWriter()
        if self.profiler is True:
            self.schedule_wait = 100
            self.schedule_warmup = 1
            self.schedule_active = 3
            self.schedule_repeat = 0
            if self.log_dir is not None:
                self.trace_handler = tensorboard_trace_handler(self.log_dir)
            else:
                self.trace_handler = None
            self.record_shapes = True
            self.with_stack = True
            self.profile_memory = True
            if 'schedule_wait' in kwargs.keys():
                self.schedule_wait = kwargs['schedule_wait']
            if 'schedule_warmup' in kwargs.keys():
                self.schedule_warmup = kwargs['schedule_warmup']
            if 'schedule_active' in kwargs.keys():
                self.schedule_active = kwargs['schedule_active']
            if 'schedule_repeat' in kwargs.keys():
                self.schedule_repeat = kwargs['schedule_repeat']
            if 'trace_handler' in kwargs.keys():
                self.trace_handler = kwargs['trace_handler']
            if 'record_shapes' in kwargs.keys():
                self.record_shapes = kwargs['record_shapes']
            if "with_stack" in kwargs.keys():
                self.with_stack = kwargs['with_stack']
            if 'profile_memory' in kwargs.keys():
                self.profile_memory = kwargs['profile_memory']
            self.prof = profile(
                schedule=schedule(wait=self.schedule_wait, warmup=self.schedule_warmup,
                                  active=self.schedule_active, repeat=self.schedule_repeat),
                on_trace_ready=self.trace_handler,
                record_shapes=self.record_shapes,
                with_stack=self.with_stack,
                profile_memory=self.profile_memory)
        else:
            if ('schedule_wait' in kwargs.keys()) or ('schedule_warmup' in kwargs.keys()) or ('schedule_active' in kwargs.keys()) or ('schedule_repeat' in kwargs.keys()) or ('trace_handler' in kwargs.keys()) or ('record_shapes' in kwargs.keys()) or ("with_stack" in kwargs.keys()) or ('profile_memory' in kwargs.keys()):
                warnings.warn(
                    'Pytorch profiler not set! Set profiler = True first!')

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

    def get_workers(self) -> int:
        return cpu_count()

    def get_profiler(self) -> profile:
        try:
            return self.profiler
        except AttributeError:
            raise RuntimeError('pytorch profiler not set!')
