"""Tee stdout into a numbered line buffer the web page can poll.

The planner, executor and skills report everything with print(); the web
front-end shows those lines instead of rewriting them all to a logger.
"""
import sys
import threading
import time


class LogBuffer:
    def __init__(self, maxlen=5000):
        self._lock = threading.Lock()
        self._lines = []          # [(seq, "HH:MM:SS", text)]
        self._next = 0
        self._partial = ""
        self.maxlen = maxlen

    @classmethod
    def install(cls):
        buf = cls()
        sys.stdout = _Tee(sys.stdout, buf)
        return buf

    @property
    def next_seq(self):
        with self._lock:
            return self._next

    def write(self, text):
        with self._lock:
            self._partial += text
            *done, self._partial = self._partial.split("\n")
            now = time.strftime("%H:%M:%S")
            for line in done:
                self._lines.append((self._next, now, line))
                self._next += 1
            if len(self._lines) > self.maxlen:
                del self._lines[:len(self._lines) - self.maxlen]

    def since(self, seq):
        with self._lock:
            return [{"seq": s, "t": ts, "text": t} for s, ts, t in self._lines if s >= seq]


class _Tee:
    def __init__(self, stream, buf):
        self._stream = stream
        self._buf = buf

    def write(self, text):
        self._buf.write(text)
        return self._stream.write(text)

    def flush(self):
        self._stream.flush()

    def __getattr__(self, name):
        return getattr(self._stream, name)
