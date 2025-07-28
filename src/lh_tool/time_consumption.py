# -*- encoding: utf-8 -*-
"""
@File    :   time_consumption.py
@Time    :   2024/03/14 10:53:51
@Author  :   lihao57
@Version :   1.0
@Contact :   lihao57@baidu.com
"""


import time
from typing import Callable, Optional
import functools


class TimeConsumptionDecorator:
    """
    TimeConsumptionDecorator used to print the module time consumption (Deprecated)

    Parameters:
        print_func (callable): The print function used to print the time consumption, default is `print`

    Example:
        ```python
        @TimeConsumptionDecorator()
        def test1():
            time.sleep(1)

        test1()

        # using logging to print
        import logging
        @TimeConsumptionDecorator(logging.info)
        def test2():
            time.sleep(1)

        test2()
        ```
    """

    def __init__(self, print_func: Callable = print):
        self._print_func = print_func

    def __call__(self, func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            """wrapper"""
            start_time = time.time()
            ret = func(*args, **kwargs)
            end_time = time.time()
            self._print_func("{} time consuming: {}".format(str(func).split(" ")[1], end_time - start_time))
            return ret

        wrapper.__name__ = func.__name__
        return wrapper


class TimeConsumptionContextManager:
    """
    TimeConsumptionContextManager used to print the module time consumption (Deprecated)

    Parameters:
        context (str): The context used to print the time consumption, default is ''
        print_func (callable): The print function used to print the time consumption, default is `print`

    Example:
        ```python
        import logging
        with TimeConsumptionContextManager('block1'):
            time.sleep(1)

        # using logging to print
        with TimeConsumptionContextManager('block2', logging.error):
            time.sleep(1)
        ```
    """

    def __init__(self, context: str = "", print_func: Callable = print):
        self._context = context
        self._print_func = print_func

    def __enter__(self):
        self.start_time = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_time = time.time()
        if self._context:
            self._print_func("{} time consuming: {}".format(self._context, self.end_time - self.start_time))
        else:
            self._print_func("time consuming: {}".format(self.end_time - self.start_time))


class TimeConsumption:
    """
    TimeConsumption used to print the module time consumption

    Parameters:
        context (str): The context used to print the time consumption, default is ''
        print_func (callable): The print function used to print the time consumption, default is `print`
        format_func (Optional[Callable]): The format function used to format the output string, default is None
        sync (bool): Whether to synchronize the cuda before and after the execution of the code block, default is False

    Example:
        ```python
        # used as decorator
        @TimeConsumption()
        def test():
            time.sleep(1)

        test()

        # used as context manager
        with TimeConsumption("block"):
            time.sleep(1)

        # for pytorch code block
        with TimeConsumption("block", sync=True):
            x = torch.randn((1000, 1000)).to("cuda")
            y = torch.randn((1000, 1000)).to("cuda")
            z = x + y
        ```
    """

    def __init__(
        self,
        context: str = "",
        print_func: Callable = print,
        format_func: Optional[Callable] = None,
        sync: bool = False,
    ):
        self._context = context
        self._print_func = print_func
        self._format_func = format_func
        self._sync = sync
        if self._sync:
            from torch.cuda import synchronize

            self._synchronize = synchronize
        else:
            self._synchronize = lambda: None

    def __call__(self, func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            """wrapper"""
            start_time = time.time()
            self._synchronize()
            ret = func(*args, **kwargs)
            self._synchronize()
            delta_time = time.time() - start_time
            if self._format_func is not None:
                delta_time = self._format_func(delta_time)
            context = self._context if self._context else str(func).split(" ")[1]
            self._print_func(f"{context} time consuming: {delta_time}")
            return ret

        wrapper.__name__ = func.__name__
        return wrapper

    def __enter__(self):
        self.start_time = time.time()
        self._synchronize()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._synchronize()
        delta_time = time.time() - self.start_time
        if self._format_func is not None:
            delta_time = self._format_func(delta_time)
        if self._context:
            self._print_func(f"{self._context} time consuming: {delta_time}")
        else:
            self._print_func(f"time consuming: {delta_time}")


def time_consumption(
    context: str = "",
    print_func: Callable = print,
    format_func: Optional[Callable] = None,
    sync: bool = False,
):
    """
    `time_consumption` used to print the module time consumption

    Parameters:
        context (str): The context used to print the time consumption, default is ''
        print_func (callable): The print function used to print the time consumption, default is `print`
        format_func (Optional[Callable]): The format function used to format the output string, default is None
        sync (bool): Whether to synchronize the cuda before and after the execution of the code block, default is False
    Example:
        ```python
        # used as decorator
        @time_consumption()
        def test():
            time.sleep(1)

        test()

        # used as context manager
        with time_consumption("block"):
            time.sleep(1)

        # for pytorch code block
        with time_consumption("block", sync=True):
            x = torch.randn((1000, 1000)).to("cuda")
            y = torch.randn((1000, 1000)).to("cuda")
            z = x + y
        ```
    """
    return TimeConsumption(context=context, print_func=print_func, format_func=format_func, sync=sync)
