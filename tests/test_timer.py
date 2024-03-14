# -*- encoding: utf-8 -*-
"""
@File    :   test_timer.py
@Time    :   2024/03/14 13:29:57
@Author  :   lihao57
@Version :   1.0
@Contact :   lihao57@baidu.com
"""

import unittest
import time
import numpy as np
import sys

sys.path.append("../src")
from lh_tool.timer import Timer


class TestTimer(unittest.TestCase):
    """TestTimer"""

    @staticmethod
    def callback(times):
        """callback"""
        times.append(time.time())

    def test_timer(self):
        """test timer"""
        times = [time.time()]
        timer = Timer(100, self.callback, (times,))
        timer.start()
        time.sleep(1)
        timer.stop()
        times = np.array(times)
        times = times[1:] - times[:-1]
        error = np.abs(times - 0.1)
        print(error)
        # 误差小于1ms
        self.assertTrue(error.max() < 0.001)


if __name__ == "__main__":
    unittest.main()
