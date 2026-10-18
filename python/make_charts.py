import re
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CSV_PATH = "paper_trades.csv"
LOG_PATH = "live_demo_output.log"
OUT_DIR = "charts"
os.makedirs(OUT_DIR, exist_ok=True)

plt.rcParams.update({
    "figure.figsize": (11, 5.5),
    "figure.dpi": 140,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "font.size": 11,
})

# ---------------- load and reconstruct cycles ----------------
df = pd.read_csv(CSV_PATH)
symbols = list(dict.fromkeys(df["symbol"]))   # preserves original order
n = len(symbols)
print(f"loaded {len(df)} rows, {n} symbols: {symbols}")

usable = (len(df) // n) * n
if usable != len(df):
    print(f"note: trimming {len(df)-usable} trailing partial-cycle row(s)")
df = df.iloc[:usable].reset_index(drop=True)

df["cycle"] = df.index // n
n_cycles = df["cycle"].nunique()
print(f"reconstructed {n_cycles} complete rebalance cycles")

# running share count per symbol
df["position"] = df.groupby("symbol")["shares_delta"].cumsum()

# elapsed seconds from start
t0 = df["ts"].iloc[0]
df["elapsed_s"] = df["ts"] - t0

# ---------------- chart 1: portfolio value ----------------
rows = []
for cyc, g in df.groupby("cycle"):
    cash_end = g["cash"].iloc[-1]
    mkt_val = (g["position"] * g["price"]).sum()
    rows.append((g["elapsed_s"].iloc[-1], cash_end + mkt_val))
pv = pd.DataFrame(rows, columns=["elapsed_s", "portfolio_value"])

fig, ax = plt.subplots()
ax.plot(pv["elapsed_s"], pv["portfolio_value"], linewidth=1.1, color="#1f5fa8")
ax.axhline(1_000_000, color="grey", linestyle="--", linewidth=1, label="Starting capital ($1,000,000)")
ax.set_title("Simulated Portfolio Value Over Live Session")
ax.set_xlabel("Elapsed time (seconds)")
ax.set_ylabel("Portfolio value (USD)")
ax.legend()
ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, p: f"${v:,.0f}"))
fig.tight_layout()
fig.savefig(f"{OUT_DIR}/1_portfolio_value.png")
plt.close(fig)

# ---------------- chart 2: weight evolution ----------------
# pick the 4 symbols whose weights vary most (most interesting to show)
variability = df.groupby("symbol")["target_w"].std().sort_values(ascending=False)
picks = list(variability.index[:4])

fig, ax = plt.subplots()
for s in picks:
    sub = df[df["symbol"] == s]
    ax.plot(sub["elapsed_s"], sub["target_w"], linewidth=0.9, label=s.upper())
ax.set_title("Solver Weight Evolution (4 most variable assets)")
ax.set_xlabel("Elapsed time (seconds)")
ax.set_ylabel("Normalized target weight")
ax.axhline(0, color="black", linewidth=0.8)
ax.legend(ncol=4, fontsize=9)
fig.tight_layout()
fig.savefig(f"{OUT_DIR}/2_weight_evolution.png")
plt.close(fig)

# ---------------- chart 3: saturation analysis ----------------
# per-symbol share of cycles at full saturation vs graduated
gross = df.groupby("cycle")["target_w"].apply(lambda x: x.abs().sum())
scale = df["cycle"].map(gross).replace(0, np.nan)
df["w_raw"] = df["target_w"] / scale * df["cycle"].map(gross)  # recover pre-normalization magnitude
df["w_unit"] = df["target_w"] / df["target_w"].abs().groupby(df["cycle"]).transform("max")

sat = df.groupby("symbol")["w_unit"].apply(lambda x: (x.abs() > 0.999).mean() * 100)
sat = sat.reindex(symbols)

fig, ax = plt.subplots()
colors = ["#c0392b" if v > 80 else "#27ae60" for v in sat.values]
ax.bar([s.upper().replace("USDT", "") for s in sat.index], sat.values, color=colors)
ax.set_title("Oscillator Saturation Rate by Asset\n(red = fully saturated ±1, green = graduated positions)")
ax.set_xlabel("Asset")
ax.set_ylabel("% of cycles at full saturation")
ax.set_ylim(0, 105)
fig.tight_layout()
fig.savefig(f"{OUT_DIR}/3_saturation_by_asset.png")
plt.close(fig)

# ---------------- chart 4: latency from log ----------------
if os.path.exists(LOG_PATH):
    with open(LOG_PATH) as f:
        lat = [float(m) for m in re.findall(r"latency_us=([\d.]+)", f.read())]
    if lat:
        lat = np.array(lat)
        fig, ax = plt.subplots()
        ax.hist(lat, bins=60, color="#2e7d32", edgecolor="white", linewidth=0.4)
        ax.axvline(np.median(lat), color="#c0392b", linestyle="--", linewidth=1.5,
                   label=f"Median = {np.median(lat):.0f} us")
        ax.axvline(np.percentile(lat, 99), color="#e67e22", linestyle="--", linewidth=1.5,
                   label=f"p99 = {np.percentile(lat,99):.0f} us")
        ax.set_title(f"End-to-End Solver Latency Distribution  (n = {len(lat)} cycles)")
        ax.set_xlabel("Latency (microseconds)")
        ax.set_ylabel("Cycle count")
        ax.legend()
        fig.tight_layout()
        fig.savefig(f"{OUT_DIR}/4_latency_histogram.png")
        plt.close(fig)
        print(f"\nlatency: median={np.median(lat):.1f}us  mean={lat.mean():.1f}us  "
              f"p99={np.percentile(lat,99):.1f}us  max={lat.max():.1f}us")
    else:
        print("no latency_us entries found in log")
else:
    print(f"{LOG_PATH} not found, skipping latency chart")

print(f"\ncharts written to {OUT_DIR}/")
for f in sorted(os.listdir(OUT_DIR)):
    print("  ", f)
