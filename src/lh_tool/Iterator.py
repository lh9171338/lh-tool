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
        raise NotImplementedError


class SingleProcess(Iterator):
    def __init__(self, process, total=None, **kwargs):
        super().__init__(process, total)

    def run(self, *args):
        # parse args
        self.parse_arg(*args)

        # run
        ret_list = []
        for args in tqdm.tqdm(zip(*self.args), total=self.total, desc=self.process.__name__):
            ret_list.append(self.process(*args))
        return ret_list


class MultiProcess(Iterator):
    def __init__(self, process, total=None, nprocs=multiprocessing.cpu_count()):
        super().__init__(process, total)

        self.nprocs = nprocs if nprocs > 0 else multiprocessing.cpu_count()

    def __call__(self, args):
        return self.process(*args)

    def run(self, *args):
        # parse args
        self.parse_arg(*args)

        # run
        with multiprocessing.Pool(self.nprocs) as p:
            ret_list = list(tqdm.tqdm(p.imap(self, zip(*self.args)), total=self.total, desc=self.process.__name__))
        return ret_list
