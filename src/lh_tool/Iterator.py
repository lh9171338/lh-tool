# -*- encoding: utf-8 -*-
"""
@File    :   iterator.py
@Time    :   2023/03/14 10:00:03
@Author  :   lihao57
@Version :   1.0
@Contact :   lihao57@baidu.com
"""

import multiprocessing
from concurrent.futures import ThreadPoolExecutor
import tqdm
import tqdm.asyncio
import functools
import asyncio
import aiomultiprocess
import numpy as np


class Iterator:
    """
    Iterator

    Parameters:
        func (callable): function to be iterated
        total (int, optional): number of iterations
    """

    def __init__(self, func, total=None, **kwargs):
        self.func = func
        self.total = total
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

        assert (
            self.total is not None
        ), "Can not infer the number of expected iterations"

        # parse args
        self.dynamic_args = [[] for _ in range(self.total)]
        self.static_args = []
        for arg in args:
            if isinstance(arg, list):
                assert len(arg) == self.total, f"{len(arg)} != {self.total}"
                for i, val in enumerate(arg):
                    self.dynamic_args[i].append(val)
            else:
                self.static_args.append(arg)

        # parse kwargs
        self.dynamic_kwargs = [{} for _ in range(self.total)]
        self.static_kwargs = {}
        for key, value in kwargs.items():
            if isinstance(value, list):
                assert (
                    len(value) == self.total
                ), f"{len(value)} != {self.total}"
                for i, val in enumerate(value):
                    self.dynamic_kwargs[i][key] = val
            else:
                self.static_kwargs[key] = value

        # fix the static args
        self.partial_func = functools.partial(
            self.func, *self.static_args, **self.static_kwargs
        )

    def run(self, *args, **kwargs):
        """run"""
        raise NotImplementedError


class SingleProcess(Iterator):
    """
    SingleProcess

    Parameters:
        func (callable): function to be iterated
        total (int, optional): number of iterations

    Example:
        .. code-block:: python
        def add(a, b):
            return a + b

        a = [1, 2]
        b = [3, 4]
        res = SingleProcess(add).run(a, b)
        print(res)
        # [4, 6]
    """

    def __init__(self, func, total=None, **kwargs):
        super().__init__(func, total)

    def run(self, *args, **kwargs):
        """run - Please ensure that static arguments precede dynamic arguments"""

        # parse
        self.parse(*args, **kwargs)

        # run
        ret_list = []
        for args, kwargs in tqdm.tqdm(
            zip(self.dynamic_args, self.dynamic_kwargs),
            total=self.total,
            desc=self.func.__name__,
        ):
            ret_list.append(self.partial_func(*args, **kwargs))
        return ret_list


class MultiProcess(Iterator):
    """
    MultiProcess

    Parameters:
        func (callable): function to be iterated
        total (int, optional): number of iterations
        nprocs (int, optional): number of processes, default is `multiprocessing.cpu_count()`

    Example:
        .. code-block:: python
        def add(a, b):
            return a + b

        a = [1, 2]
        b = [3, 4]
        res = MultiProcess(add).run(a, b)
        print(res)
        # [4, 6]
    """

    def __init__(
        self, func, total=None, nprocs=multiprocessing.cpu_count(), **kwargs
    ):
        super().__init__(func, total)

        self.nprocs = nprocs if nprocs > 0 else multiprocessing.cpu_count()

    def __call__(self, args):
        return self.partial_func(*args[0], **args[1])

    def run(self, *args, **kwargs):
        """run - Please ensure that static arguments precede dynamic arguments"""

        # parse
        self.parse(*args, **kwargs)

        # run
        with multiprocessing.Pool(self.nprocs) as p:
            ret_list = list(
                tqdm.tqdm(
                    p.imap(self, zip(self.dynamic_args, self.dynamic_kwargs)),
                    total=self.total,
                    desc=self.func.__name__,
                )
            )
        return ret_list


class AsyncProcess(Iterator):
    """
    AsyncProcess

    Parameters:
        func (callable): function to be iterated
        total (int, optional): number of iterations
        concurrency (int, optional): concurrent, default is 0

    Example:
        .. code-block:: python
        async def add(a, b):
            await asyncio.sleep(1)
            return a + b

        a = [1, 2]
        b = [3, 4]
        res = AsyncProcess(add).run(a, b)
        print(res)
        # [4, 6]
    """

    def __init__(self, func, total=None, concurrency=0, **kwargs):
        super().__init__(func, total)

        self.concurrency = concurrency

    async def __call__(self, sem, args):
        if sem is None:
            return await self.partial_func(*args[0], **args[1])
        else:
            async with sem:
                return await self.partial_func(*args[0], **args[1])

    async def main(self):
        if self.concurrency > 0:
            sem = asyncio.Semaphore(self.concurrency)
        else:
            sem = None
        tasks = [
            asyncio.create_task(self(sem, args))
            for args in zip(self.dynamic_args, self.dynamic_kwargs)
        ]
        for f in tqdm.asyncio.tqdm.as_completed(
            tasks, total=self.total, desc=self.func.__name__
        ):
            await f
        self.ret_list = [task.result() for task in tasks]

    def run(self, *args, **kwargs):
        """run - Please ensure that static arguments precede dynamic arguments"""

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
        total (int, optional): number of iterations
        concurrency (int, optional): concurrent, default is 16
        nprocs (int, optional): number of processes, default is `multiprocessing.cpu_count()`

    Example:
        .. code-block:: python
        async def add(a, b):
            await asyncio.sleep(1)
            return a + b

        a = [1, 2]
        b = [3, 4]
        res = AsyncMultiProcess(add).run(a, b)
        print(res)
        # [4, 6]
    """

    def __init__(
        self,
        func,
        total=None,
        concurrency=16,
        nprocs=multiprocessing.cpu_count(),
        **kwargs,
    ):
        super().__init__(func, total)

        self.concurrency = concurrency
        self.nprocs = nprocs

    def __call__(self, args):
        return self.partial_func(*args[0], **args[1])

    async def main(self):
        self.ret_list = []
        async with aiomultiprocess.Pool(
            self.nprocs, childconcurrency=self.concurrency
        ) as p:
            it = p.map(
                self, zip(self.dynamic_args, self.dynamic_kwargs)
            ).__aiter__()
            self.ret_list = [
                a
                async for a in tqdm.asyncio.tqdm(
                    it, total=self.total, desc=self.func.__name__
                )
            ]

    def run(self, *args, **kwargs):
        """run - Please ensure that static arguments precede dynamic arguments"""

        # parse
        self.parse(*args, **kwargs)

        # run
        asyncio.run(self.main())
        return self.ret_list


class MultiThread(Iterator):
    """
    MultiThread

    Parameters:
        func (callable): function to be iterated
        total (int, optional): number of iterations
        nworkers (int, optional): number of workers, default is 2

    Example:
        .. code-block:: python
        def add(a, b):
            return a + b

        a = [1, 2]
        b = [3, 4]
        res = MultiThread(add).run(a, b)
        print(res)
        # [4, 6]
    """

    def __init__(self, func, total=None, nworkers=2, **kwargs):
        super().__init__(func, total)

        self.nworkers = nworkers if nworkers > 0 else 2

    def __call__(self, args):
        return self.partial_func(*args[0], **args[1])

    def run(self, *args, **kwargs):
        """run - Please ensure that static arguments precede dynamic arguments"""

        # parse
        self.parse(*args, **kwargs)

        # run
        with ThreadPoolExecutor(max_workers=self.nworkers) as p:
            tasks = [
                p.submit(self, args)
                for args in zip(self.dynamic_args, self.dynamic_kwargs)
            ]
        ret_list = []
        for task in tqdm.tqdm(
            tasks, total=self.total, desc=self.func.__name__
        ):
            ret_list.append(task.result())

        return ret_list


class ParallelProcess(Iterator):
    """
    ParallelProcess

    Parameters:
        func (callable): function to be iterated
        total (int, optional): number of iterations
        nprocs (int, optional): number of processes, default is `multiprocessing.cpu_count()`

    Example:
        .. code-block:: python
        def add(arr1, arr2):
            return [a + b for a, b in zip(arr1, arr2)]

        a = [1, 2, 3, 4]
        b = [3, 4, 5, 6]
        res = ParallelProcess(add, nprocs=2).run(a, b)
        print(res)
        # [[4, 6], [8, 10]]
    """

    def __init__(
        self, func, total=None, nprocs=multiprocessing.cpu_count(), **kwargs
    ):
        super().__init__(func, total)

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

        assert (
            self.total is not None
        ), "Can not infer the number of expected iterations"

        # split tasks
        indices = np.linspace(0, self.total, self.nprocs + 1).astype("int32")

        # parse args
        self.dynamic_args = [[] for _ in range(self.nprocs)]
        self.static_args = []
        for arg in args:
            if isinstance(arg, list):
                assert (
                    len(arg) == self.total or len(arg) == self.nprocs
                ), f"{len(arg)} != {self.total} or {len(arg)} != {self.nprocs}"
                if len(arg) == self.nprocs:
                    for i in range(self.nprocs):
                        self.dynamic_args[i].append(arg[i])
                else:
                    for i in range(self.nprocs):
                        self.dynamic_args[i].append(
                            arg[indices[i] : indices[i + 1]]
                        )
            else:
                self.static_args.append(arg)

        # parse kwargs
        self.dynamic_kwargs = [{} for _ in range(self.nprocs)]
        self.static_kwargs = {}
        for key, value in kwargs.items():
            if isinstance(value, list):
                assert (
                    len(value) == self.total or len(value) == self.nprocs
                ), f"{len(value)} != {self.total} or {len(value)} != {self.nprocs}"
                if len(value) == self.nprocs:
                    for i in range(self.nprocs):
                        self.dynamic_kwargs[i][key] = value[i]
                else:
                    for i in range(self.nprocs):
                        self.dynamic_kwargs[i][key] = value[
                            indices[i] : indices[i + 1]
                        ]
            else:
                self.static_kwargs[key] = value

        # fix the static args
        self.partial_func = functools.partial(
            self.func, *self.static_args, **self.static_kwargs
        )

    def __call__(self, dynamic_args, dynamic_kwargs, results_dict, idx):
        res = self.partial_func(*dynamic_args, **dynamic_kwargs)
        results_dict[idx] = res

    def run(self, *args, **kwargs):
        """run - Please ensure that static arguments precede dynamic arguments"""

        # parse
        self.parse(*args, **kwargs)

        # run
        procs = []
        manager = multiprocessing.Manager()
        results_dict = manager.dict()
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

        return ret_list
