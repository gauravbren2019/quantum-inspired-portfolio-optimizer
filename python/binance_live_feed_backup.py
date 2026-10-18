import asyncio
import json
import time
import numpy as np
import websockets
from shared_mem_bridge import SharedMemBridge

SYMBOLS = ["btcusdt", "ethusdt", "bnbusdt", "solusdt", "xrpusdt",
           "adausdt", "dogeusdt", "avaxusdt", "linkusdt", "dotusdt"]

WINDOW = 500
REBUILD_MS = 200

class LiveFeed:
    def __init__(self, symbols):
        self.symbols = symbols
        self.n = len(symbols)
        self.prices = {s: [] for s in symbols}
        self.last_price = {s: None for s in symbols}
        self.trade_count = 0
        self.bridge = SharedMemBridge(self.n)

    def on_trade(self, symbol, price):
        buf = self.prices[symbol]
        buf.append(price)
        if len(buf) > WINDOW:
            buf.pop(0)
        self.last_price[symbol] = price
        self.trade_count += 1

    def build_moments(self):
        rets = []
        for s in self.symbols:
            buf = self.prices[s]
            if len(buf) < 10:
                return None, None
            arr = np.array(buf)
            rets.append(np.diff(arr) / arr[:-1])
        min_len = min(len(r) for r in rets)
        if min_len < 5:
            return None, None
        ret_matrix = np.vstack([r[-min_len:] for r in rets])
        mu = ret_matrix.mean(axis=1)
        sigma = np.cov(ret_matrix)
        return mu, sigma

    async def stream_symbol(self, symbol):
        url = f"wss://stream.binance.com:9443/ws/{symbol}@trade"
        while True:
            try:
                async with websockets.connect(url, open_timeout=10) as ws:
                    print(f"connected to {symbol}")
                    async for raw in ws:
                        data = json.loads(raw)
                        price = float(data["p"])
                        self.on_trade(symbol, price)
            except Exception as e:
                print(f"{symbol} connection issue: {e}, retrying in 3s")
                await asyncio.sleep(3)

    async def rebuild_loop(self):
        print("rebuild loop started, waiting for enough ticks to accumulate...")
        waiting_printed = False
        while True:
            mu, sigma = self.build_moments()
            if mu is None:
                if not waiting_printed:
                    print(f"waiting for data... trades received so far: {self.trade_count}")
                    waiting_printed = True
                await asyncio.sleep(REBUILD_MS / 1000.0)
                continue

            waiting_printed = False
            self.bridge.write_moments(mu, sigma)
            weights, latency_us = self.bridge.read_weights(timeout=1.0)
            if weights is not None:
                print(f"latency_us={latency_us:.1f} weights={np.round(weights, 4)}")
            else:
                print("no response from live_engine, is it running? (check Terminal B)")
            await asyncio.sleep(REBUILD_MS / 1000.0)

    async def run(self):
        tasks = [self.stream_symbol(s) for s in self.symbols]
        tasks.append(self.rebuild_loop())
        await asyncio.gather(*tasks)

if __name__ == "__main__":
    feed = LiveFeed(SYMBOLS)
    asyncio.run(feed.run())
