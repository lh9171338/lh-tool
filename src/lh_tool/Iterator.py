import asyncio
import functools
import multiprocessing

import tqdm
import tqdm.asyncio


class Iterator:
    """
    Iterator

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

        # infer the number of expected iterations from kwargs
        if self.total is None:
            for _, value in kwargs.items():
                if isinstance(value, list):
                    self.total = len(value)

        assert self.total is not None, 'Can not infer the number of expected iterations'

        # parse args
        self.dynamic_args = [[] for _ in range(self.total)]
        self.static_args = []
        for arg in args:
            if isinstance(arg, list):
                assert len(arg) == self.total, f'{len(arg)} != {self.total}'
                for i, val in enumerate(arg):
                    self.dynamic_args[i].append(val)
            else:
                self.static_args.append(arg)

        # parse kwargs
        self.dynamic_kwargs = [{} for _ in range(self.total)]
        self.static_kwargs = {}
        for key, value in kwargs.items():
            if isinstance(value, list):
                assert len(value) == self.total, f'{len(value)} != {self.total}'
                for i, val in enumerate(value):
                    self.dynamic_kwargs[i][key] = val
            else:
                self.static_kwargs[key] = value

        # fix the static args
        self.partial_func = functools.partial(self.func, *self.static_args, **self.static_kwargs)

    def run(self, *args, **kwargs):
        """run"""
        raise NotImplementedError


class SingleProcess(Iterator):
    """
    SingleProcess

    """

    def __init__(self, process, total=None, **kwargs):
        super().__init__(process, total)

    def run(self, *args, **kwargs):
        """run - Please ensure that static arguments precede dynamic arguments"""

        # parse
        self.parse(*args, **kwargs)

        # run
        ret_list = []
        for args, kwargs in tqdm.tqdm(zip(self.dynamic_args, self.dynamic_kwargs),
                                      total=self.total, desc=self.func.__name__):
            ret_list.append(self.partial_func(*args, **kwargs))
        return ret_list


class MultiProcess(Iterator):
    """
    MultiProcess

    """

    def __init__(self, process, total=None, nprocs=multiprocessing.cpu_count(), **kwargs):
        super().__init__(process, total)

        self.nprocs = nprocs if nprocs > 0 else multiprocessing.cpu_count()

    def __call__(self, args):
        return self.partial_func(*args[0], **args[1])

    def run(self, *args, **kwargs):
        """run - Please ensure that static arguments precede dynamic arguments"""

        # parse
        self.parse(*args, **kwargs)

        # run
        with multiprocessing.Pool(self.nprocs) as p:
            ret_list = list(tqdm.tqdm(p.imap(self, zip(self.dynamic_args, self.dynamic_kwargs)),
                                      total=self.total, desc=self.func.__name__))
        return ret_list


class AsyncProcess(Iterator):
    """
    AsyncProcess

    """

    def __init__(self, process, total=None, **kwargs):
        super().__init__(process, total)

        self.ret_list = None

    def __call__(self, args):
        return self.partial_func(*args[0], **args[1])

    async def main(self, loop):
        tasks = [loop.create_task(self(args)) for args in zip(self.dynamic_args, self.dynamic_kwargs)]
        for f in tqdm.asyncio.tqdm.as_completed(tasks, total=self.total, desc=self.func.__name__):
            await f
        self.ret_list = [task.result() for task in tasks]

    def run(self, *args, **kwargs):
        """run - Please ensure that static arguments precede dynamic arguments"""

        # parse
        self.parse(*args, **kwargs)

        # run
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(self.main(loop))
        return self.ret_list
