"""Static PDF figures for the manuscript, redrawn from the measured tables
(not exported from the interactive HTML report). Grayscale-safe: every series
is distinguished by marker/linestyle, not only by color, since a print
reviewer may see this in black and white.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linewidth": 0.5,
    "figure.dpi": 150,
})

OUT = "."

# ---------------------------------------------------------------- Figure 1
# Energy scaling law (Table 1)
models = ["Qwen2.5-1.5B", "Qwen2.5-3B", "Phi-3.5-mini", "Mistral-7B-v0.2",
          "Qwen2.5-7B", "Llama-3.1-8B", "Qwen2.5-14B"]
params = np.array([1.5, 3.0, 3.8, 7.0, 7.0, 8.0, 14.0])
j_per_tok = np.array([0.602, 1.238, 1.721, 2.842, 2.847, 3.103, 6.671])
is_qwen = np.array([1, 1, 0, 0, 1, 0, 1], dtype=bool)

fig, ax = plt.subplots(figsize=(5.2, 4.0))
ax.scatter(params[is_qwen], j_per_tok[is_qwen], marker="o", s=55,
           facecolors="none", edgecolors="black", linewidths=1.3,
           label="Qwen2.5 ladder", zorder=3)
ax.scatter(params[~is_qwen], j_per_tok[~is_qwen], marker="^", s=55,
           color="black", label="other families", zorder=3)
xs = np.linspace(1, 15, 100)
ax.plot(xs, 0.42 * xs ** 1.08 / xs, linestyle="--", color="gray", linewidth=1,
        zorder=1)  # placeholder overwritten below with a proper power fit
# proper least-squares power-law fit through the data (not forced through origin
# in log-log space, matches the ~1.08 exponent reported in the text)
logp, logj = np.log(params), np.log(j_per_tok)
b, loga = np.polyfit(logp, logj, 1)
fit_j = np.exp(loga) * xs ** b
ax.lines[-1].remove()
ax.plot(xs, fit_j, linestyle="--", color="gray", linewidth=1.2, zorder=1,
        label=f"power-law fit ($\\propto P^{{{b:.2f}}}$)")
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel("Parameters (billions)")
ax.set_ylabel("Energy per output token (J)")
ax.legend(frameon=False, loc="upper left", fontsize=8.5)
ax.set_xticks([1.5, 3, 7, 14])
ax.set_yticks([0.5, 1, 2, 4, 8])
ax.get_xaxis().set_major_formatter(mticker.ScalarFormatter())
ax.get_yaxis().set_major_formatter(mticker.ScalarFormatter())
ax.set_ylim(0.45, 9)
fig.tight_layout()
fig.savefig(f"{OUT}/fig_scaling.pdf")
plt.close(fig)

# ---------------------------------------------------------------- Figure 2
# CoT accuracy delta by scale (Table 3)
sizes = ["1.5B", "3B", "7B", "14B"]
delta = [-11.7, 0.8, -2.5, 22.5]
pvals = [0.034, 1.00, 0.73, 7.4e-6]
sig = [p < 0.05 for p in pvals]

fig, ax = plt.subplots(figsize=(5.6, 3.4))
colors = ["black" if s else "0.85" for s in sig]
hatches = [None if s else "//" for s in sig]
bars = ax.barh(sizes, delta, color=colors, edgecolor="black", height=0.55)
for bar, h in zip(bars, hatches):
    if h:
        bar.set_hatch(h)
        bar.set_facecolor("white")
ax.axvline(0, color="black", linewidth=0.8)
ax.set_xlim(-16, 30)
# Labels sit just to the right of zero, at a fixed x position, regardless of
# bar direction -- this keeps every label inside the axes and clear of the
# y-axis category ticks, unlike anchoring to each bar's own far end.
label_x = 24
for i, (d, p, s) in enumerate(zip(delta, pvals, sig)):
    label = f"{d:+.1f}  (p={p:.2g})" if s else f"{d:+.1f}  (n.s.)"
    ax.text(label_x, i, label, va="center", ha="left", fontsize=8.5)
ax.set_xlabel("Accuracy change from chain-of-thought (percentage points)")
ax.set_ylabel("Model size")
fig.tight_layout()
fig.savefig(f"{OUT}/fig_cot_scale.pdf")
plt.close(fig)

# ---------------------------------------------------------------- Figure 3
# Idle share collapse with request volume (Table 7)
rates = [100, 1000, 10000, 100000, 1000000]
idle_share_14b = [99.9, 99.3, 93.1, 57.3, 11.8]

fig, ax = plt.subplots(figsize=(5.2, 3.6))
ax.plot(rates, idle_share_14b, marker="o", color="black", linewidth=1.5,
        markersize=6)
ax.fill_between(rates, idle_share_14b, color="0.85", zorder=0)
ax.set_xscale("log")
ax.set_xlabel("Requests per day")
ax.set_ylabel("Idle power's share of total energy (%)")
ax.set_ylim(0, 105)
for r, s in zip(rates, idle_share_14b):
    # keep labels off the plotted line: high points sit above, low points below
    dy = 10 if s > 50 else -14
    ax.annotate(f"{s:.1f}%", (r, s), textcoords="offset points",
                xytext=(0, dy), ha="center", fontsize=8.5,
                bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none",
                          alpha=0.85))
fig.tight_layout()
fig.savefig(f"{OUT}/fig_idle_share.pdf")
plt.close(fig)

# ---------------------------------------------------------------- Figure 4
# Placement: joules per correct decision (Table 9)
rates2 = [100, 1000, 10000, 100000, 1000000]
edge = [27027, 2743, 315, 72, 48]
centre = [150331, 15138, 1618, 266, 131]
shared = [131, 131, 131, 131, 131]

fig, ax = plt.subplots(figsize=(5.4, 4.0))
ax.plot(rates2, edge, marker="o", linestyle="-", color="black",
        label="Edge (dedicated)", linewidth=1.5)
ax.plot(rates2, centre, marker="^", linestyle="--", color="black",
        label="Datacenter (dedicated)", linewidth=1.5)
ax.plot(rates2, shared, marker="s", linestyle=":", color="0.4",
        label="Datacenter, shared (1M/day)", linewidth=1.5)
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel("Requests per day")
ax.set_ylabel("Joules per correct decision")
ax.legend(frameon=False, loc="upper right", fontsize=8.5)
fig.tight_layout()
fig.savefig(f"{OUT}/fig_placement.pdf")
plt.close(fig)

# ---------------------------------------------------------------- Figure 5
# Daily grid carbon intensity (EPIAS, 32-day mean by hour)
hourly = [366.6, 376.2, 384.4, 389.5, 396.2, 402.8, 411.9, 406.0, 389.0, 386.2,
          385.9, 384.9, 384.0, 378.0, 366.9, 357.2, 348.9, 350.5, 354.2, 353.0,
          350.7, 355.4, 360.6, 363.8]
hours = list(range(24))

fig, ax = plt.subplots(figsize=(5.4, 3.6))
ax.plot(hours, hourly, color="black", linewidth=1.5)
best_h, worst_h = 16, 6
ax.scatter([best_h], [hourly[best_h]], color="black", zorder=5, s=60,
           marker="o")
ax.scatter([worst_h], [hourly[worst_h]], color="black", zorder=5, s=60,
           marker="^")
ax.annotate("cleanest\n16:00", (best_h, hourly[best_h]),
            textcoords="offset points", xytext=(18, 8), fontsize=8.5,
            ha="left")
ax.annotate("dirtiest\n06:00", (worst_h, hourly[worst_h]),
            textcoords="offset points", xytext=(-8, 10), fontsize=8.5,
            ha="right")
ax.set_ylim(min(hourly) - 25, max(hourly) + 15)
ax.set_xlabel("Hour of day")
ax.set_ylabel(r"Mean grid carbon intensity (gCO$_2$eq/kWh)")
ax.set_xticks([0, 6, 12, 18, 23])
ax.set_xticklabels(["00:00", "06:00", "12:00", "18:00", "23:00"])
fig.tight_layout()
fig.savefig(f"{OUT}/fig_carbon.pdf")
plt.close(fig)

print("wrote 5 figures: fig_scaling.pdf, fig_cot_scale.pdf, fig_idle_share.pdf, "
      "fig_placement.pdf, fig_carbon.pdf")
