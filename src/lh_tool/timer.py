# -*- encoding: utf-8 -*-
"""
@File    :   timer.py
@Time    :   2024/03/09 17:04:14
@Author  :   lihao57
@Version :   1.0
@Contact :   lihao57@baidu.com
"""


import threading
import time
from functools import partial
from typing import Callable, Iterable, Mapping


class Timer:
    """
    Timer

    Parameters:
        interval (int): interval time (unit: ms), default 1000ms
        callback (callable): callback function
        args (Optinal): callback args
        kwargs (Optinal): callback kwargs

    Example:
        ```python
        start_time = time.time()

        def callback():
            print(int((time.time() - start_time) * 1000))
            time.sleep(0.05)

        timer = Timer(100, callback)
        timer.start()
        time.sleep(1)
        timer.stop()
        ```
    """

    def __init__(
        self,
        interval: int = 1000,
        callback: Callable = None,
        args: Iterable = (),
        kwargs: Mapping = {},
    ):
        assert interval > 0, "interval must be greater than 0"
        assert isinstance(callback, Callable), "callback must be callable"
        self.interval = interval
        self.callback = partial(callback, *args, **kwargs)

        self.lock = threading.Lock()
        self._set_running(False)

    def start(self, session=None):
        """start timer"""
        assert not self.is_running(), "timer is start, can not start"
        self._set_running(True)
        self.thread = threading.Thread(target=self._run)
        if session is not None:
            session.register_thread(self.thread)
        self.thread.start()

    def stop(self):
        """stop timer"""
        assert self.is_running(), "timer is not start, can not stop"
        self._set_running(False)
        self.thread.join()

    def is_running(self):
        """check timer is running or not"""
        self.lock.acquire()
        is_running = self._is_running
        self.lock.release()
        return is_running

    def _set_running(self, is_running: bool):
        """set running status"""
        self.lock.acquire()
        self._is_running = is_running
        self.lock.release()

    def set_interval(self, interval: int):
        """set interval"""
        assert interval > 0, "interval must be greater than 0"
        assert not self.is_running(), "timer is start, can not set interval"
        self.interval = interval

    def _run(self):
        """run timer"""
        start_time = int(time.time() * 1000)
        cnt = 0
        while True:
            if not self.is_running():  # timer is stop
                break

            cnt += 1
            now_time = int(time.time() * 1000)
            sleep_time = cnt * self.interval - (now_time - start_time)
            if sleep_time > 0:
                time.sleep(sleep_time / 1000)

            # run callback
            self.callback()
