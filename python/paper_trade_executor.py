import time
import csv

class PaperExecutor:
    def __init__(self, symbols, starting_cash=1_000_000.0, log_path="paper_trades.csv", max_gross_leverage=1.0):
        self.symbols = symbols
        self.cash = starting_cash
        self.positions = {s: 0.0 for s in symbols}
        self.log_path = log_path
        self.max_gross_leverage = max_gross_leverage
        with open(self.log_path, "w", newline="") as f:
            csv.writer(f).writerow(["ts", "symbol", "target_w", "price", "shares_delta", "cash"])

    def _normalize_weights(self, weights):
        gross = sum(abs(w) for w in weights)
        if gross > self.max_gross_leverage and gross > 0:
            scale = self.max_gross_leverage / gross
            return [w * scale for w in weights]
        return weights

    def rebalance(self, weights, prices):
        weights = self._normalize_weights(weights)
        portfolio_value = self.cash + sum(self.positions[s] * prices[s] for s in self.symbols)

        with open(self.log_path, "a", newline="") as f:
            writer = csv.writer(f)
            for s, w in zip(self.symbols, weights):
                p = prices[s]
                target_value = w * portfolio_value
                target_shares = target_value / p if p > 0 else 0
                delta = target_shares - self.positions[s]
                self.cash -= delta * p
                self.positions[s] = target_shares
                writer.writerow([time.time(), s, round(w, 4), p, delta, self.cash])
