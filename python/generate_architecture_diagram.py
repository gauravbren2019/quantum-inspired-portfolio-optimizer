import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

fig, ax = plt.subplots(figsize=(14, 9))
ax.set_xlim(0, 14)
ax.set_ylim(0, 10)
ax.axis('off')

def box(x, y, w, h, text, facecolor, fontsize=10, fontweight='bold', textcolor='black'):
    b = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.08",
                        linewidth=1.5, edgecolor='black', facecolor=facecolor)
    ax.add_patch(b)
    ax.text(x + w/2, y + h/2, text, ha='center', va='center',
            fontsize=fontsize, fontweight=fontweight, color=textcolor, wrap=True)

def arrow(x1, y1, x2, y2):
    a = FancyArrowPatch((x1, y1), (x2, y2), arrowstyle='-|>', mutation_scale=20,
                         linewidth=2, color='#333333')
    ax.add_patch(a)

# Title
ax.text(7, 9.5, "Quantum-Inspired Simulated Bifurcation: System Architecture",
        ha='center', fontsize=15, fontweight='bold')

# Data source
box(0.5, 7.5, 3, 1.2, "Binance WebSocket\n(Live Market Data)", "#FFE8B3")
arrow(2, 7.5, 2, 6.7)

# Shared memory bridge
box(0.5, 5.5, 3, 1.2, "POSIX Shared Memory\n(Zero-Copy IPC)", "#B3D9FF")
arrow(2, 5.5, 2, 4.7)

# Preprocessing
box(0.5, 3.5, 3, 1.2, "Risk & Spectral\nPreprocessing", "#C9F7C5")
arrow(3.5, 4.1, 4.5, 4.1)

# Core engine - big box
box(4.7, 2.0, 5.0, 4.0, "", "#F3E8FF")
ax.text(7.2, 5.6, "C++20 bSB Optimization Core", ha='center', fontsize=12, fontweight='bold')
box(5.0, 4.6, 2.1, 0.8, "RMT Covariance\nFiltering", "#E0C8FF", fontsize=8)
box(7.3, 4.6, 2.1, 0.8, "Active-Set\nScreening 500->30", "#E0C8FF", fontsize=8)
box(5.0, 3.6, 2.1, 0.8, "Simulated\nBifurcation Solver", "#E0C8FF", fontsize=8)
box(7.3, 3.6, 2.1, 0.8, "Dinkelbach Sharpe\nMaximizer", "#E0C8FF", fontsize=8)
box(5.0, 2.3, 2.1, 0.8, "ADMM Cardinality\nConstraint", "#E0C8FF", fontsize=8)
box(7.3, 2.3, 2.1, 0.8, "Almgren-Chriss\nCost Model", "#E0C8FF", fontsize=8)

arrow(9.7, 4.0, 10.7, 4.0)

# Output
box(10.9, 3.4, 2.6, 1.2, "Portfolio Weights\n(+1 / -1 / graduated)", "#FFD1D1")
arrow(12.2, 3.4, 12.2, 2.6)
box(10.9, 1.4, 2.6, 1.2, "Paper Trade\nExecutor -> CSV Ledger", "#FFF3B3")

plt.tight_layout()
plt.savefig('../assets/architecture_diagram.png', dpi=150, bbox_inches='tight')
print("Saved: assets/architecture_diagram.png")
