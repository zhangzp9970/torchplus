from socket import gethostname
import datetime
import os
import random
from typing import List, Optional, TypeVar
import warnings

import numpy as np
import torch
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter
from torch.profiler import *

D_torchplus = TypeVar("D_torchplus", str, List[str])


class Init(object):
    seed = None
    log_root_dir = None
    sep = None
    backup_filename = None
    tensorboard = None
    comment = None
    deterministic = None
    profiler = None
    kwargs = None
    schedule_wait = None
    schedule_warmup = None
    schedule_active = None
    schedule_repeat = None
    trace_handler = None
    record_shapes = None
    with_stack = None
    profile_memory = None

    def __init__(
        self,
        seed: int,
        log_root_dir: Optional[str] = None,
        sep: Optional[bool] = False,
        backup_filename: Optional[str] = None,
        tensorboard: Optional[bool] = False,
        comment: Optional[str] = "",
        deterministic: Optional[bool] = False,
        profiler: Optional[bool] = False,
        **kwargs
    ) -> None:
        self.seed = seed
        self.log_root_dir = log_root_dir
        self.sep = sep
        self.backup_filename = backup_filename
        self.tensorboard = tensorboard
        self.comment = comment
        self.deterministic = deterministic
        self.profiler = profiler
        self.kwargs = kwargs
        self.__parse_args()
        self.__print_comment()
        self.__set_seed()
        self.__set_dir()
        self.__set_tensorboard()
        self.__set_backup_file()
        self.__set_profiler()

    def __parse_args(self):
        for arg in self.kwargs:
            if arg == "schedule_wait":
                self.schedule_wait = self.kwargs["schedule_wait"]
            elif arg == "schedule_warmup":
                self.schedule_warmup = self.kwargs["schedule_warmup"]
            elif arg == "schedule_active":
                self.schedule_active = self.kwargs["schedule_active"]
            elif arg == "schedule_repeat":
                self.schedule_repeat = self.kwargs["schedule_repeat"]
            elif arg == "trace_handler":
                self.trace_handler = self.kwargs["trace_handler"]
            elif arg == "record_shapes":
                self.record_shapes = self.kwargs["record_shapes"]
            elif arg == "with_stack":
                self.with_stack = self.kwargs["with_stack"]
            elif arg == "profile_memory":
                self.profile_memory = self.kwargs["profile_memory"]
            else:
                warnings.warn("Unexpected arguments: " + arg)
        pass

    def __print_comment(self):
        if self.comment != "":
            print(self.comment)

    def __set_seed(self) -> None:
        random.seed(self.seed)
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed_all(self.seed)
        np.random.seed(self.seed)
        os.environ["PYTHONHASHSEED"] = str(self.seed)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        if self.deterministic:
            cudnn.benchmark = False
            cudnn.deterministic = True
            torch.use_deterministic_algorithms(True)
        else:
            cudnn.benchmark = True
            cudnn.deterministic = False
            torch.use_deterministic_algorithms(False)

    def __set_dir(self) -> None:
        if self.log_root_dir is not None:
            current_time = datetime.datetime.now().strftime("%Y%b%d_%H-%M-%S")
            suffix = current_time + "_" + gethostname() + "_" + self.comment
            if self.sep:
                self.model_dir = os.path.join(self.log_root_dir, "Model_" + suffix)
                self.log_dir = os.path.join(self.log_root_dir, "Log_" + suffix)
                os.makedirs(self.model_dir, exist_ok=True)
                os.makedirs(self.log_dir, exist_ok=True)
            else:
                self.log_dir = os.path.join(self.log_root_dir, suffix)
                os.makedirs(self.log_dir, exist_ok=True)

    def __set_tensorboard(self) -> None:
        if self.tensorboard is True:
            try:
                os.path.exists(self.log_dir)
                self.writer = SummaryWriter(log_dir=self.log_dir, comment=self.comment)
            except:
                self.writer = SummaryWriter(comment=self.comment)
                self.log_dir = self.writer.get_logdir()

    def __set_backup_file(self) -> None:
        if self.backup_filename is not None:
            try:
                os.path.exists(self.log_dir)
                with open(
                    os.path.join(self.log_dir, os.path.basename(self.backup_filename)),
                    "w",
                    encoding="utf-8",
                ) as fout:
                    with open(
                        os.path.abspath(self.backup_filename), "r", encoding="utf-8"
                    ) as fin:
                        file = fin.read()
                        fout.write(file)
            except:
                warnings.warn("Log directory is NONE, file not backuped!")

    def __set_profiler(self) -> None:
        if self.profiler is True:
            schedule_wait = 100
            schedule_warmup = 1
            schedule_active = 3
            schedule_repeat = 0
            try:
                os.path.exists(self.log_dir)
                self.trace_handler = tensorboard_trace_handler(self.log_dir)
            except:
                trace_handler = None
            record_shapes = True
            with_stack = True
            profile_memory = True
            if self.schedule_wait is not None:
                schedule_wait = self.schedule_wait
            if self.schedule_warmup is not None:
                schedule_warmup = self.schedule_warmup
            if self.schedule_active is not None:
                schedule_active = self.schedule_active
            if self.schedule_repeat is not None:
                schedule_repeat = self.schedule_repeat
            if self.trace_handler is not None:
                trace_handler = self.trace_handler
            if self.record_shapes is not None:
                record_shapes = self.record_shapes
            if self.with_stack is not None:
                with_stack = self.with_stack
            if self.profile_memory is not None:
                profile_memory = self.profile_memory
            self.profile = profile(
                schedule=schedule(
                    wait=schedule_wait,
                    warmup=schedule_warmup,
                    active=schedule_active,
                    repeat=schedule_repeat,
                ),
                on_trace_ready=trace_handler,
                record_shapes=record_shapes,
                with_stack=with_stack,
                profile_memory=profile_memory,
            )
        else:
            if (
                (self.schedule_wait is not None)
                or (self.schedule_warmup is not None)
                or (self.schedule_active is not None)
                or (self.schedule_repeat is not None)
                or (self.trace_handler is not None)
                or (self.record_shapes is not None)
                or (self.with_stack is not None)
                or (self.profile_memory is not None)
            ):
                warnings.warn("Pytorch profiler not set! Set profiler = True first!")

    def get_device(self) -> torch.device:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def get_log_dir(self) -> D_torchplus:
        try:
            return [self.log_dir, self.model_dir] if self.sep else self.log_dir
        except AttributeError:
            raise RuntimeError("log directory not set!")

    def get_writer(self) -> SummaryWriter:
        try:
            return self.writer
        except AttributeError:
            raise RuntimeError("tensorboard summary writer not set!")

    def get_profiler(self) -> profile:
        try:
            return self.profile
        except AttributeError:
            raise RuntimeError("pytorch profiler not set!")
