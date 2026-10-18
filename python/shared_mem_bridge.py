import numpy as np
import time
from multiprocessing import shared_memory

HEADER = 2
REGION_DOUBLES = 1024 * 1024  # matches region_size in C++

class SharedMemBridge:
    def __init__(self, n_symbols, name="qisb_live"):
        self.n = n_symbols
        try:
            self.shm = shared_memory.SharedMemory(name=name, create=True, size=REGION_DOUBLES * 8)
        except FileExistsError:
            self.shm = shared_memory.SharedMemory(name=name, create=False)
        self.buf = np.ndarray((REGION_DOUBLES,), dtype=np.float64, buffer=self.shm.buf)
        self.buf[0] = 0.0

    def write_moments(self, mu, sigma):
        n = self.n
        self.buf[1] = n
        self.buf[2:2 + n] = mu
        self.buf[2 + n:2 + n + n * n] = sigma.flatten()
        self.buf[0] = 1.0  # tell c++ engine data is fresh

    def read_weights(self, timeout=0.5):
        n = self.n
        out_offset = 2 + n + n * n
        start = time.time()
        while self.buf[0] != 2.0:
            if time.time() - start > timeout:
                return None, None
            time.sleep(0.0005)
        weights = np.array(self.buf[out_offset:out_offset + n])
        latency_us = self.buf[out_offset + n]
        self.buf[0] = 0.0  # reset for next cycle
        return weights, latency_us
