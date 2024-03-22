# -*- encoding: utf-8 -*-
"""
@File    :   test_time_consumption.py
@Time    :   2024/03/14 13:46:12
@Author  :   lihao57
@Version :   1.0
@Contact :   lihao57@baidu.com
"""


import unittest
import time
import sys

sys.path.append("../src")
from lh_tool.time_consumption import (
    TimeConsumptionDecorator,
    TimeConsumptionContextManager,
    TimeConsumption,
    time_consumption,
)


class TestConsumption(unittest.TestCase):
    """TestConsumption"""

    @staticmethod
    def callback(times):
        """callback"""
        times.append(time.time())

    def test_time_consumption_decorator(self):
        """test_time_consumption_decorator"""

        @TimeConsumptionDecorator()
        def func():
            """func"""
            time.sleep(0.1)

        func()
        self.assertTrue(True)

    def test_time_consumption_context_manager(self):
        """test time consumption context manager"""
        with TimeConsumptionContextManager():
            time.sleep(0.1)
        self.assertTrue(True)

    def test_time_consumption_class(self):
        """test time consumption class"""

        @TimeConsumption()
        def func():
            """func"""
            time.sleep(0.1)

        func()

        with TimeConsumption():
            time.sleep(0.1)
        self.assertTrue(True)

    def test_time_consumption_function(self):
        """test time consumption function"""

        @time_consumption()
        def func():
            """func"""
            time.sleep(0.1)

        func()

        with time_consumption():
            time.sleep(0.1)
        self.assertTrue(True)


if __name__ == "__main__":
    unittest.main()
