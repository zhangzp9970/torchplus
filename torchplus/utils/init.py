import socket
import datetime
import os
import random
from typing import List, Optional, TypeVar
import warnings

import numpy as np
import torch
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter
from multiprocessing import cpu_count
from torch.profiler import *

D_torchplus = TypeVar('D_torchplus', str, List[str])


class Init(object):
    def __init__(self, seed: int, log_root_dir: Optional[str] = None, split: Optional[bool] = False, backup_filename: Optional[str] = None, tensorboard: Optional[bool] = False, comment: Optional[str] = '', deterministic: Optional[bool] = False, profiler: Optional[bool] = False, **kwargs) -> None:
        self.seed = seed
        self.log_root_dir = log_root_dir
        self.split = split
        self.backup_filename = backup_filename
        self.tensorboard = tensorboard
        self.comment = comment
        self.deterministic = deterministic
        self.profiler = profiler
        self.kwargs = kwargs
        self.__print_comment()
        self.__set_seed()
        self.__set_dir()
        self.__set_tensorboard()
        self.__set_backup_file()
        self.__set_profiler()

    def __print_comment(self):
        if self.comment != '':
            print(self.comment)

    def __set_seed(self) -> None:
        random.seed(self.seed)
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed_all(self.seed)
        np.random.seed(self.seed)
        os.environ['PYTHONHASHSEED'] = str(self.seed)
        if self.deterministic:
            cudnn.benchmark = False
            cudnn.deterministic = True
            torch.use_deterministic_algorithms(True)

    def __set_dir(self) -> None:
        if self.log_root_dir is not None:
            current_time = datetime.datetime.now().strftime('%b%d_%H-%M-%S')
            suffix = current_time + '_' + socket.gethostname() + '_' + self.comment
            if self.split:
                self.model_dir = os.path.join(
                    self.log_root_dir, 'Model_'+suffix)
                self.log_dir = os.path.join(
                    self.log_root_dir, 'Log_'+suffix)
                os.makedirs(self.model_dir, exist_ok=True)
                os.makedirs(self.log_dir, exist_ok=True)
            else:
                self.log_dir = os.path.join(
                    self.log_root_dir, suffix)
                os.makedirs(self.log_dir, exist_ok=True)

    def __set_tensorboard(self):
        if self.tensorboard is True:
            try:
                os.path.exists(self.log_dir)
                self.writer = SummaryWriter(
                    log_dir=self.log_dir, comment=self.comment)
            except:
                self.writer = SummaryWriter(comment=self.comment)
                self.log_dir = self.writer.get_logdir()

    def __set_backup_file(self) -> None:
        if self.backup_filename is not None:
            try:
                os.path.exists(self.log_dir)
                with open(os.path.join(self.log_dir, os.path.basename(self.backup_filename)), 'w', encoding='utf-8') as fout:
                    with open(os.path.abspath(self.backup_filename), 'r', encoding='utf-8') as fin:
                        file = fin.read()
                        fout.write(file)
            except:
                warnings.warn('Log directory is NONE, file not backuped!')

    def __set_profiler(self):
        if self.profiler is True:
            self.schedule_wait = 100
            self.schedule_warmup = 1
            self.schedule_active = 3
            self.schedule_repeat = 0
            try:
                os.path.exists(self.log_dir)
                self.trace_handler = tensorboard_trace_handler(self.log_dir)
            except:
                self.trace_handler = None
            self.record_shapes = True
            self.with_stack = True
            self.profile_memory = True
            if 'schedule_wait' in self.kwargs.keys():
                self.schedule_wait = self.kwargs['schedule_wait']
            if 'schedule_warmup' in self.kwargs.keys():
                self.schedule_warmup = self.kwargs['schedule_warmup']
            if 'schedule_active' in self.kwargs.keys():
                self.schedule_active = self.kwargs['schedule_active']
            if 'schedule_repeat' in self.kwargs.keys():
                self.schedule_repeat = self.kwargs['schedule_repeat']
            if 'trace_handler' in self.kwargs.keys():
                self.trace_handler = self.kwargs['trace_handler']
            if 'record_shapes' in self.kwargs.keys():
                self.record_shapes = self.kwargs['record_shapes']
            if "with_stack" in self.kwargs.keys():
                self.with_stack = self.kwargs['with_stack']
            if 'profile_memory' in self.kwargs.keys():
                self.profile_memory = self.kwargs['profile_memory']
            self.profile = profile(
                schedule=schedule(wait=self.schedule_wait, warmup=self.schedule_warmup,
                                  active=self.schedule_active, repeat=self.schedule_repeat),
                on_trace_ready=self.trace_handler,
                record_shapes=self.record_shapes,
                with_stack=self.with_stack,
                profile_memory=self.profile_memory)
        else:
            if ('schedule_wait' in self.kwargs.keys()) or ('schedule_warmup' in self.kwargs.keys()) or ('schedule_active' in self.kwargs.keys()) or ('schedule_repeat' in self.kwargs.keys()) or ('trace_handler' in self.kwargs.keys()) or ('record_shapes' in self.kwargs.keys()) or ("with_stack" in self.kwargs.keys()) or ('profile_memory' in self.kwargs.keys()):
                warnings.warn(
                    'Pytorch profiler not set! Set profiler = True first!')

    def get_device(self) -> torch.device:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def get_log_dir(self) -> D_torchplus:
        try:
            return [self.log_dir, self.model_dir] if self.split else self.log_dir
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
            return self.profile
        except AttributeError:
            raise RuntimeError('pytorch profiler not set!')
