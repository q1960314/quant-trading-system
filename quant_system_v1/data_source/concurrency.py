"""
动态并发检测 — 根据本地 CPU 核心数 + API 限制自适应调整

规则:
- 默认并发 = min(CPU核心数, 8)
- Tushare jiaoch.site 限制: 并发=2 (硬限制)
- 东财/AKShare: 并发=4~8
- 每次API调用间隔 >= batch_io_interval (防止被封)
"""
import os
import time
import threading
from utils.logger import get_logger

logger = get_logger("concurrency")


class DynamicConcurrency:
    def __init__(self, api_name="tushare", max_concurrent=2,
                 batch_interval=10, max_rpm=200):
        self.api_name = api_name
        self.max_concurrent = max_concurrent
        self.batch_interval = batch_interval
        self.max_rpm = max_rpm
        self._semaphore = threading.BoundedSemaphore(max_concurrent)
        self._last_call = 0
        self._call_count = 0
        self._minute_start = time.time()

    @classmethod
    def detect(cls, api_name="tushare"):
        cpu_count = os.cpu_count() or 4
        if api_name == "tushare":
            return cls(api_name=api_name, max_concurrent=2, batch_interval=10, max_rpm=200)
        elif api_name in ("akshare", "eastmoney"):
            max_c = min(cpu_count, 8)
            return cls(api_name=api_name, max_concurrent=max_c, batch_interval=3, max_rpm=500)
        elif api_name == "baostock":
            return cls(api_name=api_name, max_concurrent=1, batch_interval=5, max_rpm=100)
        else:
            return cls(api_name=api_name, max_concurrent=min(cpu_count, 4), batch_interval=5, max_rpm=300)

    def acquire(self):
        self._semaphore.acquire()
        now = time.time()
        elapsed = now - self._last_call
        if elapsed < self.batch_interval / 1000.0:
            time.sleep(self.batch_interval / 1000.0 - elapsed)
        if now - self._minute_start > 60:
            self._call_count = 0
            self._minute_start = now
        if self._call_count >= self.max_rpm:
            wait = 60 - (now - self._minute_start)
            if wait > 0:
                logger.debug(f"RPM limit reached, waiting {wait:.1f}s")
                time.sleep(wait)
            self._call_count = 0
            self._minute_start = time.time()
        self._call_count += 1
        self._last_call = time.time()
        return True

    def release(self):
        self._semaphore.release()

    def __enter__(self):
        self.acquire()
        return self

    def __exit__(self, *args):
        self.release()
