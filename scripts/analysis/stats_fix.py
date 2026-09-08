"""Statistical work requested by the referee, computed from the raw result files.

Covers, in order:
  1. interaction test  -- does CoT's effect actually differ between 1.5B and 14B?
                          (comparing p=0.034 with p=7.4e-6 is not a test of that)
  2. clustering        -- same test permuted at template level, since tasks are
                          not independent: they cluster by family and template
  3. Holm vs Bonferroni on the four Table 3 comparisons
  4. Table 4          -- the twelve family x size cells, tested with Holm
  5. NetArena 7B/14B  -- paired McNemar, since both ran the same 120 tasks
  6. Table 2          -- variance components: 3 prompts x 3 reps is nested,
                          so SD/sqrt(9) is the wrong standard error
  7. MDE              -- minimum detectable effect at 80% power per design
  8. Wilson bounds    -- for the 120/120 oracle check and the solver's 48/48
"""
import json
import math
import random
import statistics as st
from itertools import combinations

random.seed(20260908)
D = "/mnt/data/joule-spike/"
SHORT = lambda r: r.split("/")[-1].replace("Qwen2.5-", "").replace("-Instruct", "")


def mcnemar_exact(b, c):
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def holm(pairs):
    """pairs: [(label, p)] -> [(label, p, p_adj, reject_at_05)]"""
    s = sorted(pairs, key=lambda x: x[1])
    m = len(s)
    out, running = [], 0.0
    for i, (lab, p) in enumerate(s):
        adj = min(1.0, max(running, (m - i) * p))
        running = adj
        out.append((lab, p, adj, adj < 0.05))
    return out


rob = json.load(open(D + "robustness_result.json"))
R = {SHORT(r["repo"]): r for r in rob["results"]}
TASKS = sorted(R["1.5B"]["cot_correct"].keys())
FAM = lambda t: t.split("_")[0]
# templates were assigned round-robin over the build order
order = {t: i for i, t in enumerate(sorted(TASKS, key=lambda x: (FAM(x), x)))}


def delta(model, t):
    return int(R[model]["cot_correct"][t]) - int(R[model]["direct_correct"][t])


print("=" * 68)
print("1. INTERACTION: does the CoT effect differ between 1.5B and 14B?")
print("=" * 68)
Dif = [delta("14B", t) - delta("1.5B", t) for t in TASKS]
obs = st.mean(Dif)
nz = [d for d in Dif if d != 0]
cnt = 0
N = 200000
for _ in range(N):
    s = sum(d if random.random() < 0.5 else -d for d in nz)
    if abs(s) >= abs(sum(nz)):
        cnt += 1
p_int = (cnt + 1) / (N + 1)
print(f"  mean paired difference of deltas (14B - 1.5B) = {obs*100:+.1f} pp")
print(f"  non-zero pairs: {len(nz)} of {len(Dif)}")
print(f"  sign-flip permutation p = {p_int:.3g}   ({N:,} permutations)")

print()
print("=" * 68)
print("2. SAME TEST, PERMUTED AT TEMPLATE LEVEL (tasks are not independent)")
print("=" * 68)
# cluster = (family, template); template = position within family block mod 3
byfam = {}
for t in TASKS:
    byfam.setdefault(FAM(t), []).append(t)
clus = {}
for f, ts in byfam.items():
    for i, t in enumerate(sorted(ts)):
        clus[t] = (f, i % 3)
groups = {}
for t in TASKS:
    groups.setdefault(clus[t], []).append(delta("14B", t) - delta("1.5B", t))
gsum = {g: sum(v) for g, v in groups.items()}
obs_tot = sum(gsum.values())
cnt = 0
for _ in range(N):
    s = sum(v if random.random() < 0.5 else -v for v in gsum.values())
    if abs(s) >= abs(obs_tot):
        cnt += 1
p_clu = (cnt + 1) / (N + 1)
print(f"  clusters (family x template): {len(gsum)}")
print(f"  cluster-level sign-flip p = {p_clu:.3g}")

print()
print("=" * 68)
print("3. TABLE 3: Holm instead of Bonferroni")
print("=" * 68)
t3 = []
for m in ("1.5B", "3B", "7B", "14B"):
    b = sum(1 for t in TASKS if R[m]["direct_correct"][t] and not R[m]["cot_correct"][t])
    c = sum(1 for t in TASKS if not R[m]["direct_correct"][t] and R[m]["cot_correct"][t])
    t3.append((m, mcnemar_exact(b, c)))
for lab, p, adj, rej in holm(t3):
    print(f"  {lab:5s} raw p={p:9.3g}   Holm-adjusted={adj:8.3g}   {'significant' if rej else 'not significant'}")

print()
print("=" * 68)
print("4. TABLE 4: twelve family x size cells, Holm-corrected")
print("=" * 68)
cells = []
for m in ("1.5B", "3B", "7B", "14B"):
    for f in ("subnet", "acl", "bgp"):
        ts = [t for t in TASKS if FAM(t) == f]
        b = sum(1 for t in ts if R[m]["direct_correct"][t] and not R[m]["cot_correct"][t])
        c = sum(1 for t in ts if not R[m]["direct_correct"][t] and R[m]["cot_correct"][t])
        d = (sum(R[m]["cot_correct"][t] for t in ts) - sum(R[m]["direct_correct"][t] for t in ts)) / len(ts)
        cells.append((f"{m:5s} {f:6s} d={d*100:+5.1f}pp b={b:2d} c={c:2d}", mcnemar_exact(b, c)))
for lab, p, adj, rej in holm(cells):
    print(f"  {lab}  raw p={p:8.3g}  Holm={adj:8.3g}  {'SIGNIFICANT' if rej else '-'}")

print()
print("=" * 68)
print("5. NETARENA 7B vs 14B: paired McNemar (same 120 tasks)")
print("=" * 68)
per = {}
for f in ("netarena_grade_result.json", "netarena_grade_result2.json"):
    try:
        d = json.load(open(D + f))
    except FileNotFoundError:
        continue
    for e in d.get("results", []):
        if "rows" not in e:
            continue
        m = SHORT(e["repo"])
        for row in e["rows"]:
            if row["cond"] != "base":
                continue
            per.setdefault(m, {})[(f, row["task"])] = bool(row["correct"])
common = sorted(set(per.get("7B", {})) & set(per.get("14B", {})))
b = sum(1 for t in common if per["7B"][t] and not per["14B"][t])
c = sum(1 for t in common if not per["7B"][t] and per["14B"][t])
print(f"  paired tasks: {len(common)}")
print(f"  7B correct & 14B wrong: b={b}    7B wrong & 14B correct: c={c}")
print(f"  exact McNemar p = {mcnemar_exact(b, c):.3g}")
print(f"  (paper reports Fisher p = 5.2e-10 on 3/120 vs 38/120)")

print()
print("=" * 68)
print("6. TABLE 2: variance components (3 prompts x 3 reps, nested)")
print("=" * 68)
lad = {}
for f in ("ladder2_result.json", "ladder3_result.json"):
    for r in json.load(open(D + f))["results"]:
        if r.get("dirty") or not r.get("j_per_tok"):
            continue
        lad[SHORT(r["repo"])] = r
print(f"  {'model':22s} {'mean':>7} {'SD9':>7} {'SD/3':>7} {'SEprompt':>9} {'n_eff':>5}")
for m, r in sorted(lad.items(), key=lambda kv: kv[1]["params_b"]):
    rows = r["rows"]
    byp = {}
    for row in rows:
        byp.setdefault(row["prompt"], []).append(row["net_j"] / row["new_tok"])
    pm = [st.mean(v) for v in byp.values()]
    sd9 = r["j_per_tok_sd"]
    se_wrong = sd9 / 3
    se_prompt = st.stdev(pm) / math.sqrt(len(pm)) if len(pm) > 1 else float("nan")
    print(f"  {m:22s} {r['j_per_tok']:7.3f} {sd9:7.3f} {se_wrong:7.3f} {se_prompt:9.3f} {len(pm):5d}")
print("  SD/3 is what the current caption implies; SEprompt is the defensible one (df=2).")

print()
print("=" * 68)
print("7. MINIMUM DETECTABLE EFFECT at 80% power (paired, two-sided 0.05)")
print("=" * 68)
def mde(n, disc_rate, trials=4000):
    for delta_pp in range(1, 40):
        hit = 0
        for _ in range(trials):
            nd = sum(1 for _ in range(n) if random.random() < disc_rate)
            if nd == 0:
                continue
            pb = 0.5 + delta_pp / 100 * n / (2 * nd) if nd else 0.5
            pb = min(0.99, max(0.01, pb))
            c = sum(1 for _ in range(nd) if random.random() < pb)
            if mcnemar_exact(nd - c, c) < 0.05:
                hit += 1
        if hit / trials >= 0.80:
            return delta_pp
    return None
for n, dr, lab in ((120, 0.32, "n=120 paired (Table 3)"),
                   (60, 0.32, "n=60 (Table 5)"),
                   (40, 0.32, "n=40 per cell (Table 4)")):
    print(f"  {lab:26s} MDE ~ {mde(n, dr)} pp")

print()
print("=" * 68)
print("8. WILSON BOUNDS for the perfect scores")
print("=" * 68)
for k, n, lab in ((120, 120, "oracle re-derivation"), (48, 48, "solver accuracy")):
    lo, hi = wilson(k, n)
    print(f"  {lab:24s} {k}/{n} = 100%   95% CI [{lo*100:.1f}, {hi*100:.1f}]")
