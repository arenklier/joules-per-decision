"""The agent-loop figure: energy climbs, accuracy does not.

Two panels on a shared model-size axis, because the claim is a contrast
between them and splitting it across two figures would hide it. The left
panel is joules per correct decision, which is the metric the rest of the
paper uses; the right is accuracy, with the single-shot arm marked so the
eye can see that no loop arm clears it.

Same conventions as make_figures.py: Okabe-Ito, and every series carries a
marker and a linestyle so nothing depends on colour.

    python scripts/analysis/make_agent_figure.py [agent_loop_result.json]
"""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

_HERE = os.path.dirname(os.path.abspath(__file__))
FIGURES = os.path.join(os.path.dirname(os.path.dirname(_HERE)),
                       "figures")


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

BLUE = "#0072B2"
ORANGE = "#E69F00"
GREEN = "#009E73"
VERMILLION = "#D55E00"

STYLE = [
    ("single",         BLUE,       "o", "-"),
    ("repair",         ORANGE,     "^", "--"),
    ("reflect",        GREEN,      "s", "-."),
    ("reflect+repair", VERMILLION, "D", ":"),
]
LABEL = {"single": "single shot",
         "repair": "repair loop",
         "reflect": "self-review",
         "reflect+repair": "self-review + repair"}


def main(path, out=None):
    out = out or os.path.join(FIGURES, "fig_agent_loop.pdf")
    d = json.load(open(path))
    res = sorted(d["results"], key=lambda e: e["params_b"])
    params = [e["params_b"] for e in res]

    fig, (axl, axr) = plt.subplots(1, 2, figsize=(7.0, 3.1))

    for arm, colour, marker, ls in STYLE:
        jc = [e[arm]["j_per_correct"] for e in res]
        ac = [e[arm]["acc"] * 100 for e in res]
        axl.plot(params, jc, color=colour, marker=marker, linestyle=ls,
                 linewidth=1.4, markersize=4.5, label=LABEL[arm], zorder=3)
        axr.plot(params, ac, color=colour, marker=marker, linestyle=ls,
                 linewidth=1.4, markersize=4.5, zorder=3)

    axl.set_xscale("log")
    axl.set_yscale("log")
    axl.set_xticks(params)
    axl.get_xaxis().set_major_formatter(mticker.ScalarFormatter())
    axl.get_yaxis().set_major_formatter(mticker.ScalarFormatter())
    axl.set_yticks([30, 60, 120, 250, 500])
    axl.set_xlabel("Parameters (billions)")
    axl.set_ylabel("Joules per correct decision")
    axl.set_title("Energy", fontsize=10, loc="left")
    axl.legend(frameon=False, fontsize=8, loc="upper left")

    axr.set_xscale("log")
    axr.set_xticks(params)
    axr.get_xaxis().set_major_formatter(mticker.ScalarFormatter())
    axr.set_xlabel("Parameters (billions)")
    axr.set_ylabel("Accuracy (\\%)" if plt.rcParams.get("text.usetex")
                   else "Accuracy (%)")
    axr.set_ylim(30, 62)
    axr.set_title("Accuracy", fontsize=10, loc="left")

    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    print("wrote %s" % out)

    # The numbers the caption will have to quote.
    print("\namplification in joules per correct decision, against single shot")
    for e in res:
        base = e["single"]["j_per_correct"]
        rel = "  ".join("%s %.2fx" % (a, e[a]["j_per_correct"] / base)
                        for a, _, _, _ in STYLE[1:])
        print("   %-6s %s" % ("%gB" % e["params_b"], rel))
    lo = min(e[a]["j_per_correct"] / e["single"]["j_per_correct"]
             for e in res for a, _, _, _ in STYLE[1:])
    hi = max(e[a]["j_per_correct"] / e["single"]["j_per_correct"]
             for e in res for a, _, _, _ in STYLE[1:])
    print("   range across all loop arms and sizes: %.2f--%.2fx" % (lo, hi))

    print("\naccuracy change against single shot, percentage points")
    for e in res:
        base = e["single"]["acc"] * 100
        print("   %-6s %s" % ("%gB" % e["params_b"], "  ".join(
            "%s %+.1f" % (a, e[a]["acc"] * 100 - base)
            for a, _, _, _ in STYLE[1:])))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "agent_loop_result.json")
