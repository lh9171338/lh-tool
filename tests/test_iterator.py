# -*- encoding: utf-8 -*-
"""
@File    :   test_iterator.py
@Time    :   2024/03/14 13:29:51
@Author  :   lihao57
@Version :   1.0
@Contact :   lihao57@baidu.com
"""


import unittest
import asyncio
import sys

sys.path.append("../src")
from lh_tool.iterator import (
    SingleProcess,
    MultiProcess,
    AutoMultiProcess,
    BoundedMultiProcess,
    AutoBoundedMultiProcess,
    MultiThread,
    AutoMultiThread,
    AsyncProcess,
    AsyncMultiProcess,
    AutoAsyncMultiProcess,
    ParallelProcess,
    AutoParallelProcess,
)


class TestIterator(unittest.TestCase):
    """Test Iterator"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.a = [i for i in range(10)]
        self.b = [i for i in range(10)]
        self.res = [i + j for i, j in zip(self.a, self.b)]

    @staticmethod
    def process(a, b, opt="+"):
        """process"""
        if opt == "+":
            return a + b
        else:
            return a - b

    @staticmethod
    def bounded_process(a, b, opt="+", port=8000):
        """resource slot process"""
        print(f"{port}: {a} {opt} {b}")
        if opt == "+":
            return a + b
        else:
            return a - b

    @staticmethod
    def process_batch(a_list, b_list, opt="+"):
        """process batch"""
        res_list = []
        for a, b in zip(a_list, b_list):
            if opt == "+":
                res = a + b
            else:
                res = a - b
            res_list.append(res)
        return res_list

    @staticmethod
    def process_batch_v2(a_list, b_list, opt="+", _counter=None):
        """process batch"""
        res_list = []
        for a, b in zip(a_list, b_list):
            if opt == "+":
                res = a + b
            else:
                res = a - b
            res_list.append(res)
            if _counter is not None:
                with _counter.get_lock():
                    _counter.value += 1
        return res_list

    @staticmethod
    async def async_process(a, b, opt="+"):
        """async process"""
        await asyncio.sleep(1)
        if opt == "+":
            return a + b
        else:
            return a - b

    def test_single_process(self):
        """test single process"""
        ret_list = SingleProcess(self.process).run(self.a, self.b, opt="+")
        self.assertEqual(ret_list, self.res)
        exception = None
        try:
            ret_list = SingleProcess(self.process).run(self.a, self.b, "+")
        except Exception as e:
            exception = e
            print(e)
        self.assertIsInstance(exception, RuntimeError)

    def test_multi_process(self):
        """test multi process"""
        ret_list = MultiProcess(self.process, nprocs=10).run(self.a, self.b, opt="+")
        self.assertEqual(ret_list, self.res)

    def test_auto_multi_process(self):
        """test auto multi process"""
        ret_list = AutoMultiProcess(self.process, nprocs=10).run(self.a, self.b, opt="+")
        self.assertEqual(ret_list, self.res)
        ret_list = AutoMultiProcess(self.process, nprocs=1).run(self.a, self.b, opt="+")
        self.assertEqual(ret_list, self.res)

    def test_bounded_multi_process(self):
        """test resource slot multi process"""
        ret_list = BoundedMultiProcess(self.bounded_process, nprocs=2).run(self.a, self.b, opt="+", port=[8000, 8001])
        self.assertEqual(ret_list, self.res)

    def test_auto_bounded_multi_process(self):
        """test auto resource slot multi process"""
        ret_list = AutoBoundedMultiProcess(self.bounded_process, nprocs=2).run(
            self.a, self.b, opt="+", port=[8000, 8001]
        )
        self.assertEqual(ret_list, self.res)
        ret_list = AutoBoundedMultiProcess(self.bounded_process, nprocs=1).run(self.a, self.b, opt="+", port=8000)
        self.assertEqual(ret_list, self.res)

    def test_multi_thread(self):
        """test multi thread"""
        ret_list = MultiThread(self.process, nworkers=2).run(self.a, self.b, opt="+")
        self.assertEqual(ret_list, self.res)

    def test_auto_multi_thread(self):
        """test auto multi thread"""
        ret_list = AutoMultiThread(self.process, nworkers=2).run(self.a, self.b, opt="+")
        self.assertEqual(ret_list, self.res)
        ret_list = AutoMultiThread(self.process, nworkers=1).run(self.a, self.b, opt="+")
        self.assertEqual(ret_list, self.res)

    def test_async_process(self):
        """test async process"""
        ret_list = AsyncProcess(self.async_process).run(self.a, self.b, opt="+")
        self.assertEqual(ret_list, self.res)

    def test_async_multi_process(self):
        """test async multi process"""
        ret_list = AsyncMultiProcess(self.async_process, nprocs=10).run(self.a, self.b, opt="+")
        self.assertEqual(ret_list, self.res)

    def test_auto_async_multi_process(self):
        """test auto async multi process"""
        ret_list = AutoAsyncMultiProcess(self.async_process, nprocs=10).run(self.a, self.b, opt="+")
        self.assertEqual(ret_list, self.res)
        ret_list = AutoAsyncMultiProcess(self.async_process, nprocs=1).run(self.a, self.b, opt="+")
        self.assertEqual(ret_list, self.res)

    def test_parallel_process(self):
        """test parallel process"""
        ret_list = ParallelProcess(self.process_batch, is_single_task_func=False).run(self.a, self.b, opt="+")
        self.assertEqual(ret_list, self.res)

        ret_list = ParallelProcess(self.process_batch_v2, is_single_task_func=False).run(self.a, self.b, opt="+")
        self.assertEqual(ret_list, self.res)

        ret_list = ParallelProcess(self.process, is_single_task_func=True).run(self.a, self.b, opt="+")
        self.assertEqual(ret_list, self.res)
        exception = None
        try:
            ret_list = ParallelProcess(self.process_batch, is_single_task_func=False).run(self.a, self.b, "+")
        except Exception as e:
            exception = e
            print(e)
        self.assertIsInstance(exception, RuntimeError)

    def test_auto_parallel_process(self):
        """test auto parallel process"""
        ret_list = AutoParallelProcess(self.process_batch, is_single_task_func=False, nprocs=2).run(
            self.a, self.b, opt="+"
        )
        self.assertEqual(ret_list, self.res)
        ret_list = AutoParallelProcess(self.process_batch, is_single_task_func=False, nprocs=1).run(
            self.a, self.b, opt="+"
        )
        self.assertEqual(ret_list, self.res)

        ret_list = AutoParallelProcess(self.process_batch_v2, is_single_task_func=False, nprocs=2).run(
            self.a, self.b, opt="+"
        )
        self.assertEqual(ret_list, self.res)
        ret_list = AutoParallelProcess(self.process_batch_v2, is_single_task_func=False, nprocs=1).run(
            self.a, self.b, opt="+"
        )
        self.assertEqual(ret_list, self.res)

        ret_list = AutoParallelProcess(self.process, is_single_task_func=True, nprocs=2).run(self.a, self.b, opt="+")
        self.assertEqual(ret_list, self.res)
        ret_list = AutoParallelProcess(self.process, is_single_task_func=True, nprocs=1).run(self.a, self.b, opt="+")
        self.assertEqual(ret_list, self.res)


if __name__ == "__main__":
    unittest.main()
