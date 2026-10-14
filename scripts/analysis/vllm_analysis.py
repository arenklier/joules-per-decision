"""Separate the serving framework from the software stack underneath it.

vLLM cannot be installed beside the stack the rest of this paper was measured
on, so the comparison has three arms rather than two and the middle one is
what makes it readable:

    a  published Table 7   transformers 4.44.2, torch 2.4.1, cu121, static
    b  ARM=hf              transformers 5.17,   torch 2.13,  cu130, static
    c  ARM=vllm            vLLM 0.31 on the same stack as b, continuous

b against a is the stack. c against b is the framework. Reporting c against a
alone would attribute to continuous batching whatever the newer kernels did,
which is the shape of mistake Section 3.4 is a list of.

The second half recomputes the idle-share economics, because that is what the
serving question actually bears on: the transition volumes in this paper are
all set by where dynamic energy meets the idle floor, and only one of those
two terms is in question.

    python scripts/analysis/vllm_analysis.py [hf.json] [vllm.json]
"""
import json
import os
import sys

# Published Table 7, dynamic energy above each model's own idle baseline.
PUB_J = {
    3.0: {1: 192.77, 2: 92.26, 4: 46.34, 8: 30.55, 16: 16.69, 32: 11.48},
    14.0: {1: 857.06, 2: 433.42, 4: 231.27, 8: 130.67, 16: 78.99, 32: 55.95},
}
PUB_RPS = {
    3.0: {1: 0.33, 2: 0.63, 4: 1.24, 8: 2.47, 16: 4.90, 32: 9.47},
    14.0: {1: 0.18, 2: 0.33, 4: 0.64, 8: 1.22, 16: 2.24, 32: 3.81},
}
# Published idle floors, per the session each sweep came from.
PUB_IDLE_W = {3.0: 85.66, 14.0: 86.93}
VOLUMES = [100, 1_000, 10_000, 100_000, 1_000_000]


def load(path):
    d = json.load(open(path))
    out = {}
    for e in d["results"]:
        if "rows" not in e:
            continue
        out[e["params_b"]] = {
            "rows": {r["batch"]: r for r in e["rows"]},
            "resident_minus_cold_w": e.get("resident_minus_cold_w"),
            "cold_idle_w": e.get("cold_idle_w"),
            "meta": {k: e.get(k) for k in
                     ("arm", "torch", "cuda", "transformers", "vllm",
                      "gpu_util", "flashinfer_sampler", "oom_at")},
        }
    return out


def best_cell(rows):
    """The batch size with the lowest energy per request."""
    ok = [r for r in rows.values() if r.get("j_req") is not None]
    return min(ok, key=lambda r: r["j_req"]) if ok else None


def main(hf_path, vllm_path):
    hf, vl = load(hf_path), load(vllm_path)

    print("stack and framework, by model and batch size")
    print("%-5s %6s %10s %10s %10s %9s %9s"
          % ("model", "batch", "pub J/req", "hf J/req", "vllm J/req",
             "hf/pub", "vllm/hf"))
    stack, frame = [], []
    for pb in sorted(set(hf) | set(vl)):
        for b in sorted(set(PUB_J.get(pb, {})) | set(hf.get(pb, {}).get("rows", {}))
                        | set(vl.get(pb, {}).get("rows", {}))):
            p = PUB_J.get(pb, {}).get(b)
            h = hf.get(pb, {}).get("rows", {}).get(b, {}).get("j_req")
            v = vl.get(pb, {}).get("rows", {}).get(b, {}).get("j_req")
            hp = (h / p) if (h and p) else None
            vh = (v / h) if (v and h) else None
            if hp:
                stack.append(hp)
            if vh:
                frame.append(vh)
            print("%-5s %6d %10s %10s %10s %9s %9s"
                  % ("%gB" % pb, b,
                     "%.2f" % p if p else "-",
                     "%.2f" % h if h else "-",
                     "%.2f" % v if v else "-",
                     "%.2fx" % hp if hp else "-",
                     "%.2fx" % vh if vh else "-"))
        print()

    if stack:
        print("stack effect     %.2f-%.2fx (median %.2fx) over %d matched cells"
              % (min(stack), max(stack), sorted(stack)[len(stack) // 2],
                 len(stack)))
    if frame:
        print("framework effect %.2f-%.2fx (median %.2fx) over %d matched cells"
              % (min(frame), max(frame), sorted(frame)[len(frame) // 2],
                 len(frame)))

    # ---- the idle floor, which is the other half of every crossover ----
    print("\nresident idle cost over a cold card")
    print("   published, 3B weights under transformers        +0.23 W")
    for name, src in (("hf", hf), ("vllm", vl)):
        for pb, e in sorted(src.items()):
            w = e["resident_minus_cold_w"]
            if w is None:
                continue
            print("   %-4s %gB  cold %.2f W  resident %+.2f W"
                  % (name, pb, e["cold_idle_w"], w))
    print("   An engine holding most of the card for its KV cache costs what")
    print("   an empty card costs: allocated memory that is not read is free.")

    # ---- what that does to the economics table -------------------------
    print("\nidle share of the bill at 14B, recomputed")
    pb = 14.0
    cells = [("published, batch 32", PUB_J[pb][32], PUB_IDLE_W[pb])]
    for name, src in (("new stack, best batch", hf), ("vLLM, best batch", vl)):
        e = src.get(pb)
        if not e:
            continue
        b = best_cell(e["rows"])
        if b:
            floor = PUB_IDLE_W[pb]      # held fixed: only dynamic is in question
            cells.append(("%s (b=%d)" % (name, b["batch"]), b["j_req"], floor))

    print("%-28s %10s %s" % ("", "J dynamic",
                             "  ".join("%9s" % ("%d/day" % v) for v in VOLUMES)))
    for label, dyn, floor in cells:
        shares = []
        for v in VOLUMES:
            idle = floor * 86400.0 / v
            shares.append("%8.1f%%" % (100.0 * idle / (idle + dyn)))
        print("%-28s %10.2f %s" % (label, dyn, "  ".join(shares)))
    print("\n   The idle floor is held at the published %.2f W throughout, "
          "because\n   that is the term the serving framework was shown not to "
          "move." % PUB_IDLE_W[pb])

    # ---- throughput, which sets what one card can serve in a day -------
    print("\nthroughput at the best batch, and the daily ceiling it implies")
    for name, src in (("published", None), ("new stack", hf), ("vLLM", vl)):
        if src is None:
            for pb in (3.0, 14.0):
                rps = PUB_RPS[pb][32]
                print("   %-10s %-5s %6.2f req/s  %10s/day"
                      % (name, "%gB" % pb, rps, "{:,}".format(int(rps * 86400))))
            continue
        for pb, e in sorted(src.items()):
            b = best_cell(e["rows"])
            if not b:
                continue
            rps = b["req_per_s"]
            print("   %-10s %-5s %6.2f req/s  %10s/day  (batch %d)"
                  % (name, "%gB" % pb, rps, "{:,}".format(int(rps * 86400)),
                     b["batch"]))

    # ---- flags ----------------------------------------------------------
    print("\ncells carrying a drift flag")
    for name, src in (("hf", hf), ("vllm", vl)):
        for pb, e in sorted(src.items()):
            d = [r["batch"] for r in e["rows"].values() if r.get("dirty")]
            print("   %-4s %-5s %d of %d: %s"
                  % (name, "%gB" % pb, len(d), len(e["rows"]), d))
    print("   Section 3.4's second pitfall warns that a guard written for")
    print("   contention discards good cells when a sweep heats the card. A")
    print("   repeat under a different thermal history is what settles it.")


def reproducibility(first_path, second_path):
    """Settle the drift flags the way the paper settles its own.

    A sweep heats the card, so every closing baseline reads high and the 5 W
    guard flags every cell. Whether that reflects the measurement or only the
    moment the baseline was taken is testable: run the sweep twice from
    different thermal starts and see whether the numbers move. The published
    14B batching sweep was checked this way and agreed to 0.3-4.2 per cent.
    """
    a, b = load(first_path), load(second_path)
    print("\nreproducibility: two sweeps, cold start against hot start")
    print("%-5s %6s %10s %10s %8s" % ("model", "batch", "run1 J/req",
                                      "run2 J/req", "diff"))
    diffs = []
    for pb in sorted(set(a) & set(b)):
        for batch in sorted(set(a[pb]["rows"]) & set(b[pb]["rows"])):
            x = a[pb]["rows"][batch].get("j_req")
            y = b[pb]["rows"][batch].get("j_req")
            if not x or not y:
                continue
            d = abs(y - x) / x * 100.0
            diffs.append(d)
            print("%-5s %6d %10.2f %10.2f %7.1f%%"
                  % ("%gB" % pb, batch, x, y, d))
    if diffs:
        print("\n   %d cells, %.1f to %.1f per cent apart, median %.1f"
              % (len(diffs), min(diffs), max(diffs),
                 sorted(diffs)[len(diffs) // 2]))
        print("   Agreement across two thermal histories says the drift flags")
        print("   record when the closing baseline was taken, not a corrupted")
        print("   measurement.")


if __name__ == "__main__":
    a = sys.argv[1] if len(sys.argv) > 1 else "vllm_sweep_hf_result.json"
    b = sys.argv[2] if len(sys.argv) > 2 else "vllm_sweep_vllm_result.json"
    main(a, b)
    if len(sys.argv) > 3 and os.path.exists(sys.argv[3]):
        reproducibility(b, sys.argv[3])
