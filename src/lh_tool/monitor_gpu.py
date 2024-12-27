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
                    logging.warning(
                        f"low utilization detected, utilization: {utilizations}"
                    )
        else:
            low_utilization_start_time = None

        if (utilizations == last_utilizations).all():
            if constant_utilization_start_time is None:
                constant_utilization_start_time = cur_time
            else:
                logging.info(f"constant utilization detected: {cur_time - constant_utilization_start_time}s")
                if cur_time - constant_utilization_start_time >= args.time_threshold:
                    trigger = True
                    logging.warning(
                        f"constant utilization detected, utilization: {utilizations}"
                    )
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
