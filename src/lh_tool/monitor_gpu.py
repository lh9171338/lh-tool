# -*- encoding: utf-8 -*-
"""
@File    :   monitor_gpu.py
@Time    :   2024/11/08 13:02:38
@Author  :   lihao57
@Version :   1.0
@Contact :   lihao57@baidu.com
"""


import os
import time
import logging
import argparse
import GPUtil
import numpy as np
import pynvml
import threading
from typing import Optional, Union, List, Callable
import functools


class GPUPeakMemoryMonitor:
    """GPU Peak Memory Monitor

    Parameters:
        context (str): The context used to print the time consumption, default is ''
        gpu_ids (Optional[Union[List[int], int]]): The ids of the GPU device that you want to monitor, default is None, meaning all GPUs
        interval (float): The interval between sampling in seconds, default is 1.0
        format_func (Optional[Callable]): The function used to format the output message, default is None
        print_func (Optional[Callable]): The function used to print the output message, default is `print`

    Example:
        ```python
        monitor = GPUPeakMemoryMonitor("block", gpu_ids=0)
        # disable print
        # monitor = GPUPeakMemoryMonitor("block", gpu_ids=0, print_func=None)
        monitor.start()
        # to do something
        peak_memories = monitor.stop()

        # used as context manager
        with GPUPeakMemoryMonitor("block", gpu_ids=None, format_func=lambda x: f"{x / (1024 ** 3):.1f}GB"):
            # to do something

        # used as decorator
        @GPUPeakMemoryMonitor()
        def test():
            # to do something

        test()
        ```
    """

    def __init__(
        self,
        context: str = "",
        gpu_ids: Optional[Union[List[int], int]] = None,
        interval: float = 1.0,
        format_func: Optional[Callable] = None,
        print_func: Optional[Callable] = print,
    ):
        if gpu_ids is None:
            import torch

            gpu_ids = list(range(torch.cuda.device_count()))
        elif isinstance(gpu_ids, int):
            gpu_ids = [gpu_ids]
        assert isinstance(gpu_ids, list), f"`gpu_ids` must be a list, but got {type(gpu_ids)}"
        assert len(gpu_ids), "`gpu_ids` cannot be empty"

        self._context = context
        self._gpu_ids = gpu_ids
        self._interval = interval
        self._format_func = format_func
        self._print_func = print_func

        self._peak_memories = [0] * len(self._gpu_ids)
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def _monitor(self):
        """monitor"""
        pynvml.nvmlInit()
        handles = [pynvml.nvmlDeviceGetHandleByIndex(gpu_id) for gpu_id in self._gpu_ids]

        try:
            while not self._stop_event.is_set():
                for i, handle in enumerate(handles):
                    mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                    current_memory = mem_info.used
                    if current_memory > self._peak_memories[i]:
                        self._peak_memories[i] = current_memory
                time.sleep(self._interval)
        finally:
            pynvml.nvmlShutdown()

    def start(self):
        """start"""
        if self._thread is not None:
            raise RuntimeError("Monitor is already running")
        self.reset()
        self._thread = threading.Thread(target=self._monitor)
        self._thread.start()

    def stop(self) -> List[int]:
        """stop"""
        if self._thread is None:
            raise RuntimeError("Monitor is not running")
        self._stop_event.set()
        self._thread.join()
        self._thread = None
        self._stop_event.clear()
        peak_memories = self.peak_memories
        if self._print_func is not None:
            if self._context:
                self._print_func(f"{self._context} peak memory usage: {peak_memories}")
            else:
                self._print_func(f"peak memory usage: {peak_memories}")
        return peak_memories

    def reset(self):
        """reset"""
        self._peak_memories = [0] * len(self._gpu_ids)

    @property
    def peak_memories(self) -> List[int]:
        """peak memories"""
        if self._format_func is not None:
            return [self._format_func(x) for x in self._peak_memories]
        else:
            return self._peak_memories

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *args):
        self.stop()

    def __call__(self, func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            """wrapper"""
            if not self._context:
                self._context = str(func).split(" ")[1]
            self.start()
            ret = func(*args, **kwargs)
            self.stop()
            return ret

        wrapper.__name__ = func.__name__
        return wrapper


def monitor(args):
    """
    Monitor gpu utilization

    Args:
        args (dict): arguments

    Returns:
        None
    """
    interval = args.interval
    assert interval > 0, "interval must be greater than 0"

    # get initial gpu utilization and memory usage
    last_utilizations = []
    gpus = GPUtil.getGPUs()
    for gpu in gpus:
        if args.gpus is not None:
            if gpu.id not in args.gpus:
                continue

        last_utilizations.append(gpu.load * 100)
    last_utilizations = np.asarray(last_utilizations)
    assert len(last_utilizations), f"no gpu found {args.gpus}"

    low_utilization_start_time = None
    constant_utilization_start_time = None
    while True:
        trigger = False
        gpus = GPUtil.getGPUs()
        utilizations = []
        for gpu in gpus:
            if args.gpus is not None:
                if gpu.id not in args.gpus:
                    continue

            utilizations.append(gpu.load * 100)
        utilizations = np.asarray(utilizations)
        cur_time = time.time()
        if (utilizations <= args.utilization_threshold).all():
            if low_utilization_start_time is None:
                low_utilization_start_time = cur_time
            else:
                logging.info(f"low utilization detected: {cur_time - low_utilization_start_time}s")
                if cur_time - low_utilization_start_time >= args.time_threshold:
                    trigger = True
                    logging.warning(f"low utilization detected, utilization: {utilizations}")
        else:
            low_utilization_start_time = None

        if (utilizations == last_utilizations).all():
            if constant_utilization_start_time is None:
                constant_utilization_start_time = cur_time
            else:
                logging.info(f"constant utilization detected: {cur_time - constant_utilization_start_time}s")
                if cur_time - constant_utilization_start_time >= args.time_threshold:
                    trigger = True
                    logging.warning(f"constant utilization detected, utilization: {utilizations}")
        else:
            constant_utilization_start_time = None
        last_utilizations = utilizations

        if trigger:
            if args.command is not None:
                logging.info(f"execute command: {args.command}")
                os.system(args.command)
                logging.info(f"command executed")

            if args.once:
                break

            low_utilization_start_time = None
            constant_utilization_start_time = None

        # sleep
        time.sleep(interval)


def main():
    # set base logging config
    fmt = "[%(asctime)s - %(levelname)s - %(filename)s:%(lineno)s] %(message)s"
    logging.basicConfig(format=fmt, level=logging.INFO)

    parser = argparse.ArgumentParser("GPU Monitor")
    parser.add_argument(
        "-g",
        "--gpus",
        type=int,
        nargs="*",
        help="gpus to be monitored, e.g., --gpus 0 1 2 3 4 5 6 8, default: None (i.e., all gpus)",
    )
    parser.add_argument(
        "-i",
        "--interval",
        type=float,
        help="monitor interval (units of second), default: 10s",
        default=10.0,
    )
    parser.add_argument(
        "-u",
        "--utilization_threshold",
        type=float,
        help="utilization threshold (units of percentage), default: 0%%",
        default=0,
    )
    parser.add_argument(
        "-t",
        "--time_threshold",
        type=float,
        help="time threshold for monitoring (units of second), default: 300s",
        default=300,
    )
    parser.add_argument(
        "-c",
        "--command",
        type=str,
        help="command to be executed after detecting abnormality, default: None",
    )
    parser.add_argument(
        "-o",
        "--once",
        action="store_true",
        help="only monitor once",
    )

    args = parser.parse_args()
    logging.info(args)

    monitor(args)


if __name__ == "__main__":
    main()
