# -*- encoding: utf-8 -*-
"""
@File    :   time_consumption.py
@Time    :   2024/03/14 10:53:51
@Author  :   lihao57
@Version :   1.0
@Contact :   lihao57@baidu.com
"""


import time
from typing import Callable


class TimeConsumptionDecorator:
    """
    TimeConsumptionDecorator used to print the module time consumption

    Parameters:
        print_func (callable): The print function used to print the time consumption, default is `print`

    Example:
        .. code-block:: python

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
    """

    def __init__(self, print_func: Callable = print):
        self.print_func = print_func

    def __call__(self, func):
        def wrapper(*args, **kwargs):
            """wrapper"""
            start_time = time.time()
            ret = func(*args, **kwargs)
            end_time = time.time()
            self.print_func(
                "{} time consuming: {}".format(
                    str(func).split(" ")[1], end_time - start_time
                )
            )
            return ret

        wrapper.__name__ = func.__name__
        return wrapper


class TimeConsumptionContextManager:
    """
    TimeConsumptionContextManager used to print the module time consumption

    Parameters:
        context (str): The context used to print the time consumption, default is ''
        print_func (callable): The print function used to print the time consumption, default is `print`

    Example:
        .. code-block:: python

        import logging
        with TimeConsumptionContextManager('block1'):
            time.sleep(1)

        # using logging to print
        with TimeConsumptionContextManager('block2', logging.error):
            time.sleep(1)

    """

    def __init__(self, context: str = "", print_func: Callable = print):
        self.context = context
        self.print_func = print_func

    def __enter__(self):
        self.start_time = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_time = time.time()
        if self.context:
            self.print_func(
                "{} time consuming: {}".format(
                    self.context, self.end_time - self.start_time
                )
            )
        else:
            self.print_func(
                "time consuming: {}".format(self.end_time - self.start_time)
            )


class TimeConsumption:
    """
    TimeConsumption used to print the module time consumption

    Parameters:
        context (str): The context used to print the time consumption, default is ''
        print_func (callable): The print function used to print the time consumption, default is `print`

    Example:
        .. code-block:: python

        # used as decorator
        @TimeConsumption()
        def test():
            time.sleep(1)

        test()

        # used as context manager
        with TimeConsumption("block"):
            time.sleep(1)
    """

    def __init__(self, context: str = "", print_func: Callable = print):
        self.context = context
        self.print_func = print_func

    def __call__(self, func):
        def wrapper(*args, **kwargs):
            """wrapper"""
            start_time = time.time()
            ret = func(*args, **kwargs)
            end_time = time.time()
            context = self.context if self.context else str(func).split(" ")[1]
            self.print_func(
                "{} time consuming: {}".format(context, end_time - start_time)
            )
            return ret

        wrapper.__name__ = func.__name__
        return wrapper

    def __enter__(self):
        self.start_time = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_time = time.time()
        if self.context:
            self.print_func(
                "{} time consuming: {}".format(
                    self.context, self.end_time - self.start_time
                )
            )
        else:
            self.print_func(
                "time consuming: {}".format(self.end_time - self.start_time)
            )


def time_consumption(context: str = "", print_func: Callable = print):
    """
    `time_consumption` used to print the module time consumption

    Parameters:
        context (str): The context used to print the time consumption, default is ''
        print_func (callable): The print function used to print the time consumption, default is `print`
    Example:
        .. code-block:: python

        # used as decorator
        @time_consumption()
        def test():
            time.sleep(1)

        test()

        # used as context manager
        with time_consumption("block"):
            time.sleep(1)
    """
    return TimeConsumption(context, print_func)
