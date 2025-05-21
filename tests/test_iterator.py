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
    MultiThread,
    AsyncProcess,
    AsyncMultiProcess,
    ParallelProcess,
)


class TestIterator(unittest.TestCase):
    """TestIterator"""

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
        result_list = SingleProcess(self.process).run(self.a, self.b, opt="+")
        self.assertEqual(result_list, self.res)
        exception = None
        try:
            result_list = SingleProcess(self.process).run(self.a, self.b, "+")
        except Exception as e:
            exception = e
            print(e)
        self.assertIsInstance(exception, RuntimeError)

    def test_multi_process(self):
        """test multi process"""
        result_list = MultiProcess(self.process, nprocs=10).run(self.a, self.b, opt="+")
        self.assertEqual(result_list, self.res)

    def test_multi_thread(self):
        """test multi thread"""
        result_list = MultiThread(self.process, nworkers=2).run(self.a, self.b, opt="+")
        self.assertEqual(result_list, self.res)

    def test_async_process(self):
        """test async process"""
        result_list = AsyncProcess(self.async_process).run(self.a, self.b, opt="+")
        self.assertEqual(result_list, self.res)

    def test_async_multi_process(self):
        """test async multi process"""
        result_list = AsyncMultiProcess(self.async_process, nprocs=10).run(self.a, self.b, opt="+")
        self.assertEqual(result_list, self.res)

    def test_parallel_process(self):
        """test parallel process"""
        ret_list = ParallelProcess(self.process_batch, is_single_task_func=False).run(self.a, self.b, opt="+")
        result_list = [_ for ret in ret_list for _ in ret]
        self.assertEqual(result_list, self.res)

        ret_list = ParallelProcess(self.process_batch_v2, is_single_task_func=False).run(self.a, self.b, opt="+")
        result_list = [_ for ret in ret_list for _ in ret]
        self.assertEqual(result_list, self.res)

        result_list = ParallelProcess(self.process, is_single_task_func=True).run(self.a, self.b, opt="+")
        self.assertEqual(result_list, self.res)
        exception = None
        try:
            ret_list = ParallelProcess(self.process_batch, is_single_task_func=False).run(self.a, self.b, "+")
        except Exception as e:
            exception = e
            print(e)
        self.assertIsInstance(exception, RuntimeError)


if __name__ == "__main__":
    unittest.main()
