import numpy as np
import csv
import os

def generate_symbol_file(path, n_events=20000, start_price=150.00, seed=0):
    rng = np.random.default_rng(seed)
    t = 34200.0
    price = start_price
    order_id = 1000000
    rows = []

    for _ in range(n_events):
        dt = rng.exponential(0.002)
        t += dt

        event_type = rng.choice([1, 2, 3, 4], p=[0.55, 0.20, 0.15, 0.10])
        direction = rng.choice([1, -1])
        size = float(rng.choice([100, 200, 300, 500, 1000]))

        tick = rng.choice([-0.01, 0.0, 0.01], p=[0.25, 0.5, 0.25])
        price = max(round(price + tick, 2), 0.01)
        price_scaled = int(round(price * 10000))

        order_id += 1
        rows.append([round(t, 9), event_type, order_id, size, price_scaled, direction])

    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(rows)

if __name__ == "__main__":
    out_dir = "../data"
    os.makedirs(out_dir, exist_ok=True)
    for i in range(500):
        generate_symbol_file(f"{out_dir}/sym_{i}_message.csv", n_events=20000, seed=i)
        if i % 50 == 0:
            print(f"generated symbol {i}")
    print("done")
