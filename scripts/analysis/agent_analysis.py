"""Turn the agent-loop run into the numbers the paper will quote.

Three things come out of here. The table of arms against model size; an exact
McNemar test on each loop arm against the single-shot baseline, since every
arm sees the same 120 tasks and the comparison is paired; and a check that the
single-shot arm reproduced the accuracies already published in Table 3, which
is the evidence that this harness measures the same thing the rest of the
paper does.

    python scripts/analysis/agent_analysis.py [agent_loop_result.json]
"""
import json
import math
import os
import sys

# Table 3's direct-condition accuracies, 120 tasks, three templates. The
# single-shot arm here runs the identical set under the identical prompts, so
# it has to land on these.
PAPER_DIRECT_ACC = {1.5: 0.3917, 3.0: 0.4833, 7.0: 0.5500, 14.0: 0.5000}

# Table 3's direct-condition energies. These are NOT expected to match. The
# published cell sweeps a 40-task template block inside one window; the arm
# here holds 120 tasks times three passes inside one window, which is far
# longer under sustained load, so the floor climbs underneath it and the
# subtraction of a baseline taken before the pass over-attributes. The offset
# is reported so it can be read as the systematic it is rather than found
# later by a referee. What the arms are compared on is the ratio between them,
# where a shared offset cancels.
PAPER_DIRECT_J = {1.5: 9.21, 3.0: 19.13, 7.0: 40.64, 14.0: 81.17}

ARM_ORDER = ["single", "repair", "reflect", "reflect+repair"]


def mcnemar_exact(b, c):
    """Two-sided exact McNemar on the discordant pairs."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / (2.0 ** n)
    return min(1.0, 2.0 * tail)


def paired(base_rows, arm_rows):
    """(gained, lost) between two arms over the same task ids."""
    a = {r["task"]: r["correct"] for r in base_rows}
    b = {r["task"]: r["correct"] for r in arm_rows}
    shared = set(a) & set(b)
    gained = sum(1 for t in shared if b[t] and not a[t])
    lost = sum(1 for t in shared if a[t] and not b[t])
    return gained, lost, len(shared)


def main(path):
    d = json.load(open(path))
    passes = d.get("passes")
    print("%s -- %d passes, k=%s, opening idle %.2f W\n"
          % (os.path.basename(path), passes, d.get("max_repairs"),
             d["open_idle"]))

    print("%-8s %-15s %6s %7s %8s %6s %11s %10s %7s"
          % ("model", "arm", "calls", "acc", "J/task", "fire",
             "J/correct", "vs single", "drift"))
    rows_out = []
    for e in d["results"]:
        if "error" in e:
            print("%-8s FAILED: %s" % (e["repo"], e["error"]))
            continue
        pb = e["params_b"]
        name = ("%gB" % pb)
        base = e.get("single")
        for arm in ARM_ORDER:
            a = e.get(arm)
            if not a:
                continue
            rel = (a["j_per_correct"] / base["j_per_correct"]
                   if base and base.get("j_per_correct") else float("nan"))
            flag = "" if a.get("deterministic", True) else "  NONDET"
            if a.get("dirty"):
                flag += "  DIRTY"
            print("%-8s %-15s %6.2f %6.1f%% %8.2f %5.1f%% %11.1f %9.2fx %6.1f W%s"
                  % (name, arm, a["calls_per_task"], a["acc"] * 100,
                     a["j_per_task"], a["check_fire_rate"] * 100,
                     a["j_per_correct"], rel, a.get("drift_w", float("nan")),
                     flag))
            rows_out.append((pb, arm, a))
        print()

    # ---- paired tests against the single-shot arm ----------------------
    print("paired accuracy tests against single-shot (exact McNemar, same tasks)")
    print("%-8s %-15s %8s %7s %6s %9s"
          % ("model", "arm", "gained", "lost", "n", "p"))
    for e in d["results"]:
        base = e.get("single")
        if not base:
            continue
        for arm in ARM_ORDER[1:]:
            a = e.get(arm)
            if not a:
                continue
            g, l, n = paired(base["rows"], a["rows"])
            p = mcnemar_exact(g, l)
            print("%-8s %-15s %8d %7d %6d %9.3f"
                  % ("%gB" % e["params_b"], arm, g, l, n, p))
    print()

    # ---- does the baseline arm reproduce the published table? ----------
    print("single-shot arm against Table 3's direct accuracies")
    worst = 0.0
    for e in d["results"]:
        a = e.get("single")
        want = PAPER_DIRECT_ACC.get(e["params_b"])
        if not a or want is None:
            continue
        gap = abs(a["acc"] - want) * 100
        worst = max(worst, gap)
        print("   %-6s measured %5.1f%%   published %5.1f%%   gap %4.1f pt  %s"
              % ("%gB" % e["params_b"], a["acc"] * 100, want * 100, gap,
                 "ok" if gap < 0.1 else "MISMATCH"))
    if worst >= 0.1:
        print("\n   The single-shot arm is meant to be the published direct "
              "condition re-run.\n   A gap here means the harness diverged and "
              "the comparison above is not safe.")
    else:
        print("\n   exact match: the loop is measured against the paper's own "
              "baseline, not a new one")

    print("\nenergy offset against Table 3 (expected, not a match)")
    offs = []
    for e in d["results"]:
        a = e.get("single")
        want = PAPER_DIRECT_J.get(e["params_b"])
        if not a or want is None or not a.get("j_per_task"):
            continue
        r = a["j_per_task"] / want
        offs.append(r)
        print("   %-6s measured %7.2f J   published %7.2f J   %+5.1f%%"
              % ("%gB" % e["params_b"], a["j_per_task"], want,
                 (r - 1) * 100))
    if len(offs) > 1:
        spread = (max(offs) - min(offs)) / (sum(offs) / len(offs)) * 100
        print("\n   mean %+.1f%%, spread %.1f%% of it. A consistent offset is a "
              "window-length\n   artefact and cancels in the arm ratios; a wide "
              "spread would not."
              % ((sum(offs) / len(offs) - 1) * 100, spread))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "agent_loop_result.json")
