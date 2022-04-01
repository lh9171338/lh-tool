import multiprocessing
import tqdm


class Iterator:
    def __init__(self, process, total=None, **kwargs):
        self.process = process
        self.total = total
        self.args = None

    @staticmethod
    def to_list(val, length):
        if isinstance(val, list):
            assert len(val) == length, f'{len(val)} != {length}'
        else:
            val = [val] * length
        return val

    def parse_arg(self, *args):
        assert len(args), 'The number of arguments can not be 0'
        self.args = []
        if self.total is None:
            assert isinstance(args[0], list), 'Can not infer the number of expected iterations'
            self.total = len(args[0])
        for arg in args:
            arg = self.to_list(arg, self.total)
            self.args.append(arg)

    def run(self, *args):
        assert 0, 'Not implemented'


class SingleProcess(Iterator):
    def __init__(self, process, total=None, **kwargs):
        super().__init__(process, total)

    def run(self, *args):
        # 解析参数
        self.parse_arg(*args)

        # 运行
        retval = []
        for args in tqdm.tqdm(zip(*self.args), total=self.total, desc=self.process.__name__):
            retval.append(self.process(*args))
        return retval


class MultiProcess(Iterator):
    def __init__(self, process, total=None, nprocs=multiprocessing.cpu_count()):
        super().__init__(process, total)

        self.nprocs = nprocs if nprocs > 0 else multiprocessing.cpu_count()

    def _process(self, args):
        return self.process(*args)

    def run(self, *args):
        # 解析参数
        self.parse_arg(*args)

        # 运行
        with multiprocessing.Pool(self.nprocs) as p:
            retval = list(tqdm.tqdm(p.imap(self._process, zip(*self.args)), total=self.total, desc=self.process.__name__))
        return retval
