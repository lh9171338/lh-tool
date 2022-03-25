import multiprocessing
import tqdm


class MultiProcessing:
    def __init__(self, process, total=None, nprocs=multiprocessing.cpu_count()):
        self.process = process
        self.total = total
        self.nprocs = nprocs

    @staticmethod
    def to_list(val, length):
        if isinstance(val, list):
            assert len(val) == length, f'{len(val)} != {length}'
        else:
            val = [val] * length
        return val

    def run(self, *args):
        assert len(args), 'The number of arguments can not be 0'
        args_ = []
        if self.total is None:
            assert isinstance(args[0], list), 'Can not infer the number of expected iterations'
            self.total = len(args[0])
        for arg in args:
            arg = self.to_list(arg, self.total)
            args_.append(arg)
        with multiprocessing.Pool(self.nprocs) as p:
            retval = list(tqdm.tqdm(p.imap(self.process, zip(*args_)), total=self.total, desc=self.process.__name__))
        return retval
