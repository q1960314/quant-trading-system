"""Rate limiter for API calls."""
import time
import threading


class RateLimiter:
    """Rate limiter: rpm calls per minute, thread-safe."""
    def __init__(self, rpm=120):
        self.interval = 60.0 / rpm  # 0.5s at 120/min
        self._last = 0
        self._lock = threading.Lock()

    def acquire(self):
        with self._lock:
            now = time.time()
            wait = self._last + self.interval - now
            if wait > 0:
                time.sleep(wait)
            self._last = time.time()
