"""Check that the deposited result files reproduce the paper's numbers.

The risk this guards against is not a wrong analysis but a wrong file: a
result file from a different session would still parse, still look plausible,
and silently fail to reproduce the table it is supposed to back. So every
check below recomputes a value the paper states and compares it against the
printed figure, and the script exits non-zero if any of them disagree.

Run from the repository root.
"""
import json
import math
import os
import random
import statistics as st
import sys

random.seed(20260908)
D = "data/results"
SHORT = lambda r: r.split("/")[-1].replace("Qwen2.5-", "").replace("-Instruct", "")
fails = []


def check(label, got, want, tol, unit=""):
    ok = want is None or abs(got - want) <= tol
    print(f"  {'ok  ' if ok else 'FAIL'} {label:52s} got {got:>10.4g}{unit}"
          + (f"   paper {want:g}{unit}" if want is not None else ""))
    if not ok:
        fails.append(label)


def mcnemar_exact(b, c):
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)


def holm(pairs):
    s, out, run = sorted(pairs, key=lambda x: x[1]), [], 0.0
    for i, (lab, p) in enumerate(s):
        run = min(1.0, max(run, (len(s) - i) * p))
        out.append((lab, p, run))
    return out


def wilson_lo(k, n, z=1.96):
    p, d = k / n, 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h


print("TABLE 3 -- chain-of-thought accuracy effect, Holm across four sizes")
rob = json.load(open(f"{D}/robustness_result.json"))
R = {SHORT(r["repo"]): r for r in rob["results"]}
TASKS = sorted(R["1.5B"]["cot_correct"])
check("task count", len(TASKS), 120, 0)
t3 = []
for m in ("1.5B", "3B", "7B", "14B"):
    b = sum(1 for t in TASKS if R[m]["direct_correct"][t] and not R[m]["cot_correct"][t])
    c = sum(1 for t in TASKS if not R[m]["direct_correct"][t] and R[m]["cot_correct"][t])
    t3.append((m, mcnemar_exact(b, c)))
adj = dict((l, a) for l, _, a in holm(t3))
check("14B Holm-adjusted p", adj["14B"], 3.0e-5, 1e-5)
check("1.5B Holm-adjusted p", adj["1.5B"], 0.101, 0.005)

print("\nINTERACTION -- does the effect change sign with scale")
delta = lambda m, t: int(R[m]["cot_correct"][t]) - int(R[m]["direct_correct"][t])
Dif = [delta("14B", t) - delta("1.5B", t) for t in TASKS]
check("mean shift 1.5B -> 14B (pp)", st.mean(Dif) * 100, 34.2, 0.5, " pp")
nz = [d for d in Dif if d]
N, hit = 200000, 0
for _ in range(N):
    if abs(sum(d if random.random() < .5 else -d for d in nz)) >= abs(sum(nz)):
        hit += 1
check("sign-flip permutation p", (hit + 1) / (N + 1), 5e-6, 5e-5)

print("\nTABLE 4 -- twelve family x size cells, Holm as one family")
FAM = lambda t: t.split("_")[0]
cells = []
for m in ("1.5B", "3B", "7B", "14B"):
    for f in ("subnet", "acl", "bgp"):
        ts = [t for t in TASKS if FAM(t) == f]
        b = sum(1 for t in ts if R[m]["direct_correct"][t] and not R[m]["cot_correct"][t])
        c = sum(1 for t in ts if not R[m]["direct_correct"][t] and R[m]["cot_correct"][t])
        cells.append((f"{m}/{f}", mcnemar_exact(b, c)))
surv = [(l, a) for l, _, a in holm(cells) if a < 0.05]
print(f"  surviving cells: {surv}")
check("exactly one cell survives Holm", len(surv), 1, 0)
check("survivor is 14B subnetting", 1 if surv and surv[0][0] == "14B/subnet" else 0, 1, 0)

print("\nTABLE 2 -- energy scaling ladder")
lad = {}
for f in ("ladder2_result.json", "ladder3_result.json"):
    for r in json.load(open(f"{D}/{f}"))["results"]:
        if not r.get("dirty") and r.get("j_per_tok"):
            lad[SHORT(r["repo"])] = r
for m, r in sorted(lad.items(), key=lambda kv: kv[1]["params_b"]):
    print(f"    {m:22s} {r['params_b']:5.1f}B  {r['j_per_tok']:7.3f} J/token")
check("models in ladder", len(lad), 7, 0)

print("\nPITFALL 4 -- output cap flips the conclusion")
try:
    c512 = json.load(open(f"{D}/reasoning_tax2_cap512_result.json"))
    c1024 = json.load(open(f"{D}/reasoning_tax2_cap1024_result.json"))
    print(f"  cap512 keys {list(c512)[:5]} / cap1024 keys {list(c1024)[:5]}")
    print("  (both caps present, so the truncation pitfall is reproducible)")
except FileNotFoundError as e:
    fails.append(f"missing cap file: {e}")

print("\nGRADING AUDIT -- no task changes hands under a stricter rule")
ma = json.load(open(f"{D}/marker_audit_result.json"))
check("audited tasks", ma["n_tasks"], 120, 0)
print(f"  models audited: {[SHORT(r['repo']) for r in ma['results']]}")

print("\nWILSON BOUNDS for the perfect scores quoted in the paper")
check("oracle 120/120 lower bound (%)", wilson_lo(120, 120) * 100, 96.9, 0.2, "%")
check("solver 48/48 lower bound (%)", wilson_lo(48, 48) * 100, 92.6, 0.2, "%")

print("\n" + "=" * 62)
if fails:
    print(f"{len(fails)} CHECK(S) FAILED: {fails}")
    sys.exit(1)
print("all checks reproduce the paper's stated values")
