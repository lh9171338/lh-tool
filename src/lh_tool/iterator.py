# -*- encoding: utf-8 -*-
"""
@File    :   iterator.py
@Time    :   2023/03/14 10:00:03
@Author  :   lihao57
@Version :   1.0
@Contact :   lihao57@baidu.com
"""

import threading
import multiprocessing
from concurrent.futures import ThreadPoolExecutor, as_completed
import tqdm
import tqdm.asyncio
import functools
import asyncio
import aiomultiprocess
import numpy as np
import time
import inspect
from typing import Callable, Optional, Type


class Iterator:
    """
    Iterator

    Parameters:
        func (callable): function to be iterated
        total (int): number of iterations, if not specified, will infer from the list-type `args` and `kwargs` of `run` method
        disable_pbar (bool): whether to disable progress bar, default is `False`
    """

    def __init__(
        self,
        func: Callable,
        total: Optional[int] = None,
        disable_pbar: bool = False,
        **kwargs,
    ):
        self.func = func
        self.total = total
        self.disable_pbar = disable_pbar
        self.dynamic_args = None
        self.static_args = None
        self.dynamic_kwargs = None
        self.static_kwargs = None
        self.partial_func = None

    def parse(self, *args, **kwargs):
        """parse"""
        # infer the number of expected iterations from args
        if self.total is None:
            for arg in args:
                if isinstance(arg, list):
                    self.total = len(arg)
                    break

        # infer the number of expected iterations from kwargs
        if self.total is None:
            for _, value in kwargs.items():
                if isinstance(value, list):
                    self.total = len(value)
                    break

        assert self.total is not None, "Can not infer the number of expected iterations"

        # parse args
        self.dynamic_args = [[] for _ in range(self.total)]
        self.static_args = []
        dynamic_arg_flags = []
        for arg in args:
            if isinstance(arg, list):
                assert len(arg) == self.total, f"{len(arg)} != {self.total}"
                for i, val in enumerate(arg):
                    self.dynamic_args[i].append(val)
                dynamic_arg_flags.append(True)
            else:
                self.static_args.append(arg)
                dynamic_arg_flags.append(False)

        # Assert static arguments precede dynamic arguments
        if len(dynamic_arg_flags) > 1:
            dynamic_arg_flags = np.array(dynamic_arg_flags)
            if (dynamic_arg_flags[:-1] > dynamic_arg_flags[1:]).any():
                msg = ["dynamic" if flag else "static" for flag in dynamic_arg_flags]
                raise RuntimeError(f"Dynamic arguments precede static arguments: {msg}")

        # parse kwargs
        self.dynamic_kwargs = [{} for _ in range(self.total)]
        self.static_kwargs = {}
        for key, value in kwargs.items():
            if isinstance(value, list):
                assert len(value) == self.total, f"{len(value)} != {self.total}"
                for i, val in enumerate(value):
                    self.dynamic_kwargs[i][key] = val
            else:
                self.static_kwargs[key] = value

        # fix the static args
        self.partial_func = functools.partial(self.func, *self.static_args, **self.static_kwargs)

    def run(self, *args, **kwargs):
        """run"""
        raise NotImplementedError


class AutoIterator(Iterator):
    """
    Auto Iterator

    Create an iterator automatically based on the input arguments.

    Parameters:
        iterator_cls (type | None): iterator class, if not specified, will not use iterator
        func (callable): function to be iterated
        total (int): number of iterations, if not specified, will infer from the list-type `args` and `kwargs` of `run` method
        **kwargs: inherited from `Iterator`
    """

    def __init__(
        self,
        iterator_cls: Optional[Type[Iterator]],
        func: Callable,
        total: Optional[int] = None,
        **kwargs,
    ):
        if iterator_cls:
            self._func = iterator_cls(func=func, total=total, **kwargs).run
        else:
            self._func = func

    def run(self, *args, **kwargs):
        """run"""
        return self._func(*args, **kwargs)


class SingleProcess(Iterator):
    """
    SingleProcess

    Parameters:
        inherited from `Iterator`

    Example:
        ```python
        def add(a, b):
            return a + b

        a = [1, 2]
        b = [3, 4]
        res = SingleProcess(add).run(a, b)
        print(res)
        # [4, 6]
        ```
    """

    def run(self, *args, **kwargs):
        """
        run
            - Please ensure that static arguments precede dynamic arguments
            - If the `total` is not specified, it will be inferred from the first list-type parameter
        """
        ret_list = []
        if "_counter" in kwargs:
            _counter = kwargs.pop("_counter")

            # parse
            self.parse(*args, **kwargs)

            # run
            for args, kwargs in zip(self.dynamic_args, self.dynamic_kwargs):
                ret_list.append(self.partial_func(*args, **kwargs))
                with _counter.get_lock():
                    _counter.value += 1
        else:
            # parse
            self.parse(*args, **kwargs)

            # run
            for args, kwargs in tqdm.tqdm(
                zip(self.dynamic_args, self.dynamic_kwargs),
                total=self.total,
                desc=self.func.__name__,
                disable=self.disable_pbar,
            ):
                ret_list.append(self.partial_func(*args, **kwargs))
        return ret_list


class MultiProcess(Iterator):
    """
    Multi-Process

    Parameters:
        func (callable): function to be iterated
        total (int): number of iterations, if not specified, will infer from the list-type `args` and `kwargs` of `run` method
        nprocs (int): number of processes, default is `multiprocessing.cpu_count()`
        **kwargs: inherited from `Iterator`

    Example:
        ```python
        def add(a, b):
            return a + b

        a = [1, 2]
        b = [3, 4]
        res = MultiProcess(add).run(a, b)
        print(res)
        # [4, 6]
        ```
    """

    def __init__(
        self,
        func: Callable,
        total: Optional[int] = None,
        nprocs: int = multiprocessing.cpu_count(),
        **kwargs,
    ):
        super().__init__(func=func, total=total, **kwargs)

        self.nprocs = nprocs if nprocs > 0 else multiprocessing.cpu_count()

    def __call__(self, args):
        idx = args[2]
        return idx, self.partial_func(*args[0], **args[1])

    def run(self, *args, **kwargs):
        """
        run
            - Please ensure that static arguments precede dynamic arguments
            - If the `total` is not specified, it will be inferred from the first list-type parameter
        """
        # parse
        self.parse(*args, **kwargs)

        # run
        ret_list = [None] * self.total
        with multiprocessing.Pool(self.nprocs) as p:
            for idx, result in tqdm.tqdm(
                p.imap_unordered(self, zip(self.dynamic_args, self.dynamic_kwargs, range(self.total))),
                total=self.total,
                desc=self.func.__name__,
                disable=self.disable_pbar,
            ):
                ret_list[idx] = result
        return ret_list


class AutoMultiProcess(AutoIterator):
    """
    Auto Multi-Process

    Create a `MultiProcess` iterator when `nprocs > 1` and a `SingleProcess` iterator otherwise.

    Parameters:
        func (callable): function to be iterated
        total (int): number of iterations, if not specified, will infer from the list-type `args` and `kwargs` of `run` method
        nprocs (int): number of processes, default is `multiprocessing.cpu_count()`
        **kwargs: inherited from `AutoIterator`

    Example:
        ```python
        def add(a, b):
            return a + b

        a = [1, 2]
        b = [3, 4]
        # Use MultiProcess
        res = AutoMultiProcess(add, nprocs=2).run(a, b)
        print(res)
        # [4, 6]

        # Use SingleProcess
        res = AutoMultiProcess(add, nprocs=1).run(a, b)
        print(res)
        # [4, 6]
        ```
    """

    def __init__(
        self,
        func: Callable,
        total: Optional[int] = None,
        nprocs: int = multiprocessing.cpu_count(),
        **kwargs,
    ):
        iterator_cls = MultiProcess if nprocs > 1 else SingleProcess
        super().__init__(
            iterator_cls=iterator_cls,
            func=func,
            total=total,
            nprocs=nprocs,
            **kwargs,
        )


class BoundedMultiProcess(Iterator):
    """
    Bounded Multi-Process

    A utility to run a function in parallel using multiple processes,
    with support for per-process resource binding (e.g., GPU ID, port).
    Make sure the number of resources equals the number of processes,
    and the resources arguments must be passed as keyword arguments.

    Parameters:
        func (callable): function to be iterated
        total (int): number of iterations, if not specified, will infer from the list-type `args` and `kwargs` of `run` method
        nprocs (int): number of processes, default is `multiprocessing.cpu_count()`
        **kwargs: inherited from `Iterator`

    Example:
        ```python
        def add(a, b, port):
            print(f"{port}: {a} + {b}")
            return a + b

        a = [1, 2, 3, 4]
        b = [5, 6, 7, 8]
        # length of `port` must be equal to `nprocs`
        res = MultiProcess(add, nprocs=2).run(a, b, port=[8000, 8001])
        print(res)
        # [6, 8, 10, 12]
        ```
    """

    def __init__(
        self,
        func: Callable,
        total: Optional[int] = None,
        nprocs: int = multiprocessing.cpu_count(),
        **kwargs,
    ):
        super().__init__(func=func, total=total, **kwargs)

        self.nprocs = nprocs if nprocs > 0 else multiprocessing.cpu_count()

    def parse(self, *args, **kwargs):
        """parse"""
        # infer the number of expected iterations from args
        if self.total is None:
            for arg in args:
                if isinstance(arg, list):
                    self.total = len(arg)
                    break

        # infer the number of expected iterations from kwargs
        if self.total is None:
            for _, value in kwargs.items():
                if isinstance(value, list):
                    self.total = len(value)
                    break

        assert self.total is not None, "Can not infer the number of expected iterations"
        assert (
            self.total != self.nprocs
        ), "Ensure that `total != nprocs`; otherwise, it will be unclear whether the argument represents a resource. Please use `MultiProcess` instead"

        # parse args
        self.dynamic_args = [[] for _ in range(self.total)]
        self.static_args = []
        dynamic_arg_flags = []
        for arg in args:
            if isinstance(arg, list):
                assert len(arg) == self.total, f"{len(arg)} != {self.total}"
                for i, val in enumerate(arg):
                    self.dynamic_args[i].append(val)
                dynamic_arg_flags.append(True)
            else:
                self.static_args.append(arg)
                dynamic_arg_flags.append(False)

        # Assert static arguments precede dynamic arguments
        if len(dynamic_arg_flags) > 1:
            dynamic_arg_flags = np.array(dynamic_arg_flags)
            if (dynamic_arg_flags[:-1] > dynamic_arg_flags[1:]).any():
                msg = ["dynamic" if flag else "static" for flag in dynamic_arg_flags]
                raise RuntimeError(f"Dynamic arguments precede static arguments: {msg}")

        # parse kwargs
        self.process_kwargs = [{} for _ in range(self.nprocs)]
        self.dynamic_kwargs = [{} for _ in range(self.total)]
        self.static_kwargs = {}
        for key, value in kwargs.items():
            if isinstance(value, list):
                if len(value) == self.nprocs:
                    for i, val in enumerate(value):
                        self.process_kwargs[i][key] = val
                else:
                    assert len(value) == self.total, f"{len(value)} != {self.total}"
                    for i, val in enumerate(value):
                        self.dynamic_kwargs[i][key] = val
            else:
                self.static_kwargs[key] = value

        # fix the static args
        self.partial_func = functools.partial(self.func, *self.static_args, **self.static_kwargs)

    def initializer(self, counter):
        """initializer"""
        # Set process index
        global _proc_idx
        with counter.get_lock():
            _proc_idx = counter.value
            counter.value += 1

    def __call__(self, args):
        # Get process index
        global _proc_idx
        idx = args[2]
        return idx, self.partial_func(*args[0], **args[1], **self.process_kwargs[_proc_idx])

    def run(self, *args, **kwargs):
        """
        run
            - Please ensure that static arguments precede dynamic arguments
            - If the `total` is not specified, it will be inferred from the first list-type parameter
        """
        # parse
        self.parse(*args, **kwargs)

        # run
        counter = multiprocessing.Value("i", 0)
        ret_list = [None] * self.total
        with multiprocessing.Pool(self.nprocs, self.initializer, (counter,)) as p:
            for idx, result in tqdm.tqdm(
                p.imap_unordered(self, zip(self.dynamic_args, self.dynamic_kwargs, range(self.total))),
                total=self.total,
                desc=self.func.__name__,
                disable=self.disable_pbar,
            ):
                ret_list[idx] = result
        return ret_list


class AutoBoundedMultiProcess(AutoIterator):
    """
    Auto Bounded Multi-Process

    Create a `BoundedMultiProcess` iterator when `nprocs > 1` and a `SingleProcess` iterator otherwise.

    Parameters:
        func (callable): function to be iterated
        total (int): number of iterations, if not specified, will infer from the list-type `args` and `kwargs` of `run` method
        nprocs (int): number of processes, default is `multiprocessing.cpu_count()`
        **kwargs: inherited from `AutoIterator`

    Example:
        ```python
        def add(a, b, port):
            print(f"{port}: {a} + {b}")
            return a + b

        a = [1, 2, 3, 4]
        b = [5, 6, 7, 8]
        # Use MultiProcess, length of `port` must be equal to `nprocs`
        res = AutoBoundedMultiProcess(add, nprocs=2).run(a, b, port=[8000, 8001])
        print(res)
        # [6, 8, 10, 12]

        # Use SingleProcess, the type of `port` must be a scalar
        res = AutoBoundedMultiProcess(add, nprocs=1).run(a, b, port=8000)
        print(res)
        # [6, 8, 10, 12]
        ```
    """

    def __init__(
        self,
        func: Callable,
        total: Optional[int] = None,
        nprocs: int = multiprocessing.cpu_count(),
        **kwargs,
    ):
        iterator_cls = BoundedMultiProcess if nprocs > 1 else SingleProcess
        super().__init__(
            iterator_cls=iterator_cls,
            func=func,
            total=total,
            nprocs=nprocs,
            **kwargs,
        )


class AsyncProcess(Iterator):
    """
    AsyncProcess

    Parameters:
        func (callable): function to be iterated
        total (int): number of iterations, if not specified, will infer from the list-type `args` and `kwargs` of `run` method
        concurrency (int): concurrent, default is 0
        **kwargs: inherited from `Iterator`

    Example:
        ```python
        async def add(a, b):
            await asyncio.sleep(1)
            return a + b

        a = [1, 2]
        b = [3, 4]
        res = AsyncProcess(add).run(a, b)
        print(res)
        # [4, 6]
        ```
    """

    def __init__(
        self,
        func: Callable,
        total: Optional[int] = None,
        concurrency: int = 0,
        **kwargs,
    ):
        super().__init__(func=func, total=total, **kwargs)

        self.concurrency = concurrency

    async def __call__(self, sem, args):
        idx = args[2]
        if sem is None:
            return idx, await self.partial_func(*args[0], **args[1])
        else:
            async with sem:
                return idx, await self.partial_func(*args[0], **args[1])

    async def main(self):
        if self.concurrency > 0:
            sem = asyncio.Semaphore(self.concurrency)
        else:
            sem = None
        tasks = [
            asyncio.create_task(self(sem, args))
            for args in zip(self.dynamic_args, self.dynamic_kwargs, range(self.total))
        ]
        self.ret_list = [None] * self.total
        for task in tqdm.asyncio.tqdm.as_completed(
            tasks, total=self.total, desc=self.func.__name__, disable=self.disable_pbar
        ):
            idx, result = await task
            self.ret_list[idx] = result

    def run(self, *args, **kwargs):
        """
        run
            - Please ensure that static arguments precede dynamic arguments
            - If the `total` is not specified, it will be inferred from the first list-type parameter
        """
        # parse
        self.parse(*args, **kwargs)

        # run
        asyncio.run(self.main())
        return self.ret_list


class AsyncMultiProcess(Iterator):
    """
    AsyncMultiProcess

    Parameters:
        func (callable): function to be iterated
        total (int): number of iterations, if not specified, will infer from the list-type `args` and `kwargs` of `run` method
        concurrency (int): concurrent, default is 16
        nprocs (int): number of processes, default is `multiprocessing.cpu_count()`
        **kwargs: inherited from `Iterator`

    Example:
        ```python
        async def add(a, b):
            await asyncio.sleep(1)
            return a + b

        a = [1, 2]
        b = [3, 4]
        res = AsyncMultiProcess(add).run(a, b)
        print(res)
        # [4, 6]
        ```
    """

    def __init__(
        self,
        func: Callable,
        total: Optional[int] = None,
        concurrency: int = 16,
        nprocs: int = multiprocessing.cpu_count(),
        **kwargs,
    ):
        super().__init__(func=func, total=total, **kwargs)

        self.concurrency = concurrency
        self.nprocs = nprocs

    async def __call__(self, args):
        return await self.partial_func(*args[0], **args[1])

    async def main(self):
        async with aiomultiprocess.Pool(self.nprocs, childconcurrency=self.concurrency) as p:
            it = p.map(self, zip(self.dynamic_args, self.dynamic_kwargs)).__aiter__()
            self.ret_list = [
                result
                async for result in tqdm.asyncio.tqdm(
                    it, total=self.total, desc=self.func.__name__, disable=self.disable_pbar
                )
            ]

    def run(self, *args, **kwargs):
        """run - Please ensure that static arguments precede dynamic arguments"""

        # parse
        self.parse(*args, **kwargs)

        # run
        asyncio.run(self.main())
        return self.ret_list


class AutoAsyncMultiProcess(AutoIterator):
    """
    Auto AsyncMultiProcess

    Create a `AsyncMultiProcess` iterator when `nprocs > 1` and a `AsyncProcess` iterator otherwise.

    Parameters:
        func (callable): function to be iterated
        total (int): number of iterations, if not specified, will infer from the list-type `args` and `kwargs` of `run` method
        nprocs (int): number of processes, default is `multiprocessing.cpu_count()`
        **kwargs: inherited from `AutoIterator`

    Example:
        ```python
        async def add(a, b):
            await asyncio.sleep(1)
            return a + b

        a = [1, 2]
        b = [3, 4]
        # Use AsyncMultiProcess
        res = AutoAsyncMultiProcess(add, nprocs=2).run(a, b)
        print(res)
        # [4, 6]

        # Use AsyncProcess
        res = AutoAsyncMultiProcess(add, nprocs=1).run(a, b)
        print(res)
        # [4, 6]
        ```
    """

    def __init__(
        self,
        func: Callable,
        total: Optional[int] = None,
        nprocs: int = multiprocessing.cpu_count(),
        **kwargs,
    ):
        iterator_cls = AsyncMultiProcess if nprocs > 1 else AsyncProcess
        super().__init__(
            iterator_cls=iterator_cls,
            func=func,
            total=total,
            nprocs=nprocs,
            **kwargs,
        )


class MultiThread(Iterator):
    """
    MultiThread

    Parameters:
        func (callable): function to be iterated
        total (int): number of iterations, if not specified, will infer from the list-type `args` and `kwargs` of `run` method
        nworkers (int): number of workers, default is 2
        **kwargs: inherited from `Iterator`

    Example:
        ```python
        def add(a, b):
            return a + b

        a = [1, 2]
        b = [3, 4]
        res = MultiThread(add).run(a, b)
        print(res)
        # [4, 6]
        ```
    """

    def __init__(
        self,
        func: Callable,
        total: Optional[int] = None,
        nworkers: int = 2,
        **kwargs,
    ):
        super().__init__(func=func, total=total, **kwargs)

        self.nworkers = nworkers if nworkers > 0 else 2

    def __call__(self, args):
        idx = args[2]
        return idx, self.partial_func(*args[0], **args[1])

    def run(self, *args, **kwargs):
        """
        run
            - Please ensure that static arguments precede dynamic arguments
            - If the `total` is not specified, it will be inferred from the first list-type parameter
        """
        # parse
        self.parse(*args, **kwargs)

        # run
        ret_list = [None] * self.total
        with ThreadPoolExecutor(max_workers=self.nworkers) as p:
            tasks = [p.submit(self, args) for args in zip(self.dynamic_args, self.dynamic_kwargs, range(self.total))]
            for task in tqdm.tqdm(
                as_completed(tasks), total=self.total, desc=self.func.__name__, disable=self.disable_pbar
            ):
                idx, result = task.result()
                ret_list[idx] = result

        return ret_list


class AutoMultiThread(AutoIterator):
    """
    Auto MultiThread

    Create a `MultiThread` iterator when `nworkers > 1` and a `SingleProcess` iterator otherwise.

    Parameters:
        func (callable): function to be iterated
        total (int): number of iterations, if not specified, will infer from the list-type `args` and `kwargs` of `run` method
        nworkers (int): number of workers, default is 2
        **kwargs: inherited from `AutoIterator`

    Example:
        ```python
        def add(a, b):
            return a + b

        a = [1, 2]
        b = [3, 4]
        # Use MultiThread
        res = AutoMultiThread(add, nworkers=2).run(a, b)
        print(res)
        # [4, 6]

        # Use SingleProcess
        res = AutoMultiThread(add, nworkers=1).run(a, b)
        print(res)
        # [4, 6]
        ```
    """

    def __init__(
        self,
        func: Callable,
        total: Optional[int] = None,
        nworkers: int = 2,
        **kwargs,
    ):
        iterator_cls = MultiThread if nworkers > 1 else SingleProcess
        super().__init__(
            iterator_cls=iterator_cls,
            func=func,
            total=total,
            nworkers=nworkers,
            **kwargs,
        )


class ParallelProcess(Iterator):
    """
    ParallelProcess

    A flexible multi-process runner designed to execute a function in parallel,
    supporting both single-task and batch-task execution. It automatically splits
    input arguments across processes, handles static vs dynamic arguments, and
    provides optional progress bar support via shared counters.

    Suitable for cases where:
    - Each process handles a batch of tasks (is_single_task_func=False)
    - Each task is processed independently (is_single_task_func=True)

    Parameters:
        func (callable): function to be iterated
        total (int | None): number of iterations, if not specified, will infer from the list-type `args` and `kwargs` of `run` method
        nprocs (int): number of processes, default is `multiprocessing.cpu_count()`
        is_single_task_func (bool): whether the function is single task, default is True (when version >= 1.11.1)
        pbar_refresh_interval (float): interval of progress bar refreshing, default is 1.0s
        flatten_result (bool): whether to flatten the result for multi-task function, default is True
        **kwargs: inherited from `Iterator`

    Example:
        ```python
        # For multi-task function
        def add(arr1, arr2):
            return [a + b for a, b in zip(arr1, arr2)]

        a = [1, 2, 3, 4]
        b = [3, 4, 5, 6]
        res = ParallelProcess(add, nprocs=2, is_single_task_func=False).run(a, b)
        print(res)
        # [[4, 6], [8, 10]]

        # use `_counter` argument to display progress bar
        def add(arr1, arr2, _counter=None):
            res = []
            for a, b in zip(arr1, arr2):
                res.append(a + b)
                with _counter.get_lock():
                    _counter.value += 1
            return res

        a = [1, 2, 3, 4]
        b = [3, 4, 5, 6]
        res = ParallelProcess(add, nprocs=2, is_single_task_func=False).run(a, b)
        print(res)
        # [[4, 6], [8, 10]]

        # For single-task function
        def add(a, b):
            return a + b

        a = [1, 2, 3, 4]
        b = [3, 4, 5, 6]
        res = ParallelProcess(add, nprocs=2, is_single_task_func=True).run(a, b)
        print(res)
        # [4, 6, 8, 10]
        ```
    """

    def __init__(
        self,
        func: Callable,
        total: Optional[int] = None,
        nprocs: int = multiprocessing.cpu_count(),
        is_single_task_func: bool = True,
        pbar_refresh_interval: float = 1.0,
        **kwargs,
    ):
        if is_single_task_func:
            func = SingleProcess(func).run
        super().__init__(func=func, total=total, **kwargs)

        self.nprocs = nprocs if nprocs > 0 else multiprocessing.cpu_count()
        self.is_single_task_func = is_single_task_func
        self.pbar_refresh_interval = pbar_refresh_interval

    def parse(self, *args, **kwargs):
        """parse"""
        # infer the number of expected iterations from args
        if self.total is None:
            for arg in args:
                if isinstance(arg, list):
                    self.total = len(arg)
                    break

        # infer the number of expected iterations from kwargs
        if self.total is None:
            for _, value in kwargs.items():
                if isinstance(value, list):
                    self.total = len(value)
                    break

        assert self.total is not None, "Can not infer the number of expected iterations"
        if self.total == self.nprocs:
            print(
                "Warning: When resource arguments are provided, please ensure that `total != nprocs`; otherwise, it will be ambiguous whether the argument represents a resource or not"
            )

        # split tasks
        indices = np.linspace(0, self.total, self.nprocs + 1).astype("int32")

        # parse args
        self.dynamic_args = [[] for _ in range(self.nprocs)]
        self.static_args = []
        dynamic_arg_flags = []
        for arg in args:
            if isinstance(arg, list):
                assert (
                    len(arg) == self.total or len(arg) == self.nprocs
                ), f"{len(arg)} != {self.total} and {len(arg)} != {self.nprocs}"
                if len(arg) == self.total:
                    for i in range(self.nprocs):
                        self.dynamic_args[i].append(arg[indices[i] : indices[i + 1]])
                else:
                    for i in range(self.nprocs):
                        self.dynamic_args[i].append(arg[i])
                dynamic_arg_flags.append(True)
            else:
                self.static_args.append(arg)
                dynamic_arg_flags.append(False)

        # Assert static arguments precede dynamic arguments
        if len(dynamic_arg_flags) > 1:
            dynamic_arg_flags = np.array(dynamic_arg_flags)
            if (dynamic_arg_flags[:-1] > dynamic_arg_flags[1:]).any():
                msg = ["dynamic" if flag else "static" for flag in dynamic_arg_flags]
                raise RuntimeError(f"Dynamic arguments precede static arguments: {msg}")

        # parse kwargs
        self.dynamic_kwargs = [{} for _ in range(self.nprocs)]
        self.static_kwargs = {}
        for key, value in kwargs.items():
            if isinstance(value, list):
                assert (
                    len(value) == self.total or len(value) == self.nprocs
                ), f"{len(value)} != {self.total} or {len(value)} != {self.nprocs}"
                if len(value) == self.total:
                    for i in range(self.nprocs):
                        self.dynamic_kwargs[i][key] = value[indices[i] : indices[i + 1]]
                else:
                    for i in range(self.nprocs):
                        self.dynamic_kwargs[i][key] = value[i]
            else:
                self.static_kwargs[key] = value

        # fix the static args
        self.partial_func = functools.partial(self.func, *self.static_args, **self.static_kwargs)

    def __call__(self, dynamic_args, dynamic_kwargs, results_dict, idx):
        res = self.partial_func(*dynamic_args, **dynamic_kwargs)
        results_dict[idx] = res

    def run(self, *args, **kwargs):
        """
        run
            - Please ensure that static arguments precede dynamic arguments
            - If the `total` is not specified, it will be inferred from the first list-type parameter
        """
        # parse
        self.parse(*args, **kwargs)

        # run
        procs = []
        manager = multiprocessing.Manager()
        results_dict = manager.dict()

        # check if function has `_counter` args
        has_counter = False
        if not self.is_single_task_func:
            params = inspect.signature(self.func).parameters
            has_counter = "_counter" in params

        if self.is_single_task_func or has_counter:
            counter = multiprocessing.Value("i", 0)
            for i in range(self.nprocs):
                p = multiprocessing.Process(
                    target=self,
                    args=(
                        self.dynamic_args[i],
                        {**self.dynamic_kwargs[i], "_counter": counter},
                        results_dict,
                        i,
                    ),
                )
                procs.append(p)
                p.start()

            # display progress bar
            with tqdm.tqdm(total=self.total, disable=self.disable_pbar) as pbar:

                def _update_pbar():
                    """update function"""
                    last_val = 0
                    while any(p.is_alive() for p in procs):
                        val = counter.value
                        delta = val - last_val
                        if delta > 0:
                            pbar.update(delta)
                            last_val = val
                        time.sleep(self.pbar_refresh_interval)
                    delta = counter.value - last_val
                    if delta > 0:
                        pbar.update(delta)

                t = threading.Thread(target=_update_pbar, daemon=True)
                t.start()

                for p in procs:
                    p.join()
                t.join()

            ret_list = [results_dict[idx] for idx in range(self.nprocs)]
        else:
            for i in range(self.nprocs):
                p = multiprocessing.Process(
                    target=self,
                    args=(
                        self.dynamic_args[i],
                        self.dynamic_kwargs[i],
                        results_dict,
                        i,
                    ),
                )
                procs.append(p)
                p.start()

            for p in procs:
                p.join()

            ret_list = [results_dict[idx] for idx in range(self.nprocs)]

        # flatten the ret_list from 2d to 1d
        ret_list = [_ for sub_ret_list in ret_list for _ in sub_ret_list]

        return ret_list


class AutoParallelProcess(AutoIterator):
    """
    Auto ParallelProcess

    Create a `ParallelProcess` iterator when `nworkers > 1` and a `SingleProcess` iterator otherwise.

    Parameters:
        func (callable): function to be iterated
        total (int): number of iterations, if not specified, will infer from the list-type `args` and `kwargs` of `run` method
        nprocs (int): number of processes, default is `multiprocessing.cpu_count()`
        is_single_task_func (bool): whether the function is single task, default is True (when version >= 1.11.1)
        **kwargs: inherited from `AutoIterator`

    Example:
        ```python
        # For multi-task function
        def add(arr1, arr2):
            return [a + b for a, b in zip(arr1, arr2)]

        a = [1, 2, 3, 4]
        b = [3, 4, 5, 6]
        # Use ParallelProcess
        res = AutoParallelProcess(add, nprocs=2, is_single_task_func=False).run(a, b)
        print(res)
        # [[4, 6], [8, 10]]

        # Use original function
        res = AutoParallelProcess(add, nprocs=1, is_single_task_func=False).run(a, b)
        print(res)
        # [[4, 6], [8, 10]]

        # For single-task function
        def add(a, b):
            return a + b

        a = [1, 2, 3, 4]
        b = [3, 4, 5, 6]
        # Use ParallelProcess
        res = AutoParallelProcess(add, nprocs=2, is_single_task_func=True).run(a, b)
        print(res)
        # [4, 6, 8, 10]

        # Use SingleProcess
        res = AutoParallelProcess(add, nprocs=1, is_single_task_func=True).run(a, b)
        print(res)
        # [4, 6, 8, 10]
        ```
    """

    def __init__(
        self,
        func: Callable,
        total: Optional[int] = None,
        nprocs: int = multiprocessing.cpu_count(),
        is_single_task_func: bool = True,
        **kwargs,
    ):
        iterator_cls = ParallelProcess if nprocs > 1 else (SingleProcess if is_single_task_func else None)
        super().__init__(
            iterator_cls=iterator_cls,
            func=func,
            total=total,
            nprocs=nprocs,
            is_single_task_func=is_single_task_func,
            **kwargs,
        )
