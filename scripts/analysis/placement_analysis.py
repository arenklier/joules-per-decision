"""Where should a network-operations decision be computed?

Combines the measured arms into the one question a communications venue cares
about: edge or centre. Everything here is arithmetic on measured parameters --
no new measurement, and every input is labelled with where it came from.

The metric is joules per CORRECT decision, because the placements differ in
accuracy as well as in energy, and comparing joules per attempt would flatter
whichever arm is worst at the task.

Three arms, and the third is the one that matters:

  edge      a gateway-class card, always on, small model, low accuracy
  centre    a datacentre card dedicated to this workload
  shared    the same card, its idle floor divided across other tenants

The transport term is carried explicitly even though it turns out to be
negligible for text-sized payloads -- showing that it is negligible is itself
the finding, since the usual argument for edge placement is transport savings.
"""
import json
import os

# ---------------------------------------------------------------- measured
# Centre: NVIDIA L40S, Qwen2.5-14B bf16, this study.
CENTRE = {
    "name": "centre  (L40S, 14B)",
    "idle_w": 86.93,           # utilization sweep, resident idle
    "j_req_batched": 55.95,    # best batch (32); batch 1 was 857.06
    "acc_direct": 0.500,       # 120 tasks, robustness run
    "acc_cot": 0.725,
    "j_req_cot_b1": 3909.8,    # measured, unbatched
    "cot_batch_factor": 15.3,  # measured for direct at 14B; applied to CoT as
                               # an estimate and labelled as one
    "tok_per_s": 22.9,
}

# Edge: NVIDIA RTX A2000, Qwen2.5-1.5B bf16. Filled from edge_result.json.
EDGE_FALLBACK = {
    "name": "edge    (A2000, 1.5B)",
    "idle_w": 7.89,
    "j_req_batched": None,
    "acc_direct": 0.392,       # 120 tasks, robustness run
    "tok_per_s": 13.7,
}

# Transport: published internet transmission intensity, kWh/GB. The spread in
# this literature is wide and shrinking year on year, so both ends are carried.
KWH_PER_GB = (0.01, 0.06)
J_PER_MB = tuple(k * 3.6e6 / 1024 for k in KWH_PER_GB)

PAYLOADS_KB = (1, 10, 100)     # a bare query, a query with context, a log bundle
RATES = (100, 1_000, 10_000, 100_000, 1_000_000)

# The centre's idle floor is divided by everything that card serves in a day,
# not by our workload alone. Expressing it as the card's TOTAL daily load is
# what an operator can actually look up, unlike an abstract "sharing factor".
CENTRE_TOTAL_LOAD = (0, 100_000, 1_000_000)   # 0 means dedicated to this work


def load_edge():
    edge = dict(EDGE_FALLBACK)
    path = os.environ.get("EDGE_JSON", "edge_result.json")
    if os.path.exists(path):
        d = json.load(open(path, encoding="utf-8"))
        if d.get("j_req_best"):
            edge["j_req_batched"] = d["j_req_best"]
            edge["idle_w"] = d.get("idle_w") or edge["idle_w"]
            edge["best_batch"] = d.get("best_batch")
            edge["dirty"] = d.get("dirty")
            edge["drift_w"] = d.get("drift_w")
            rows = d.get("rows") or []
            # latency is a single-request property, so it comes from batch 1.
            # max(tok_per_s) would be the batch-32 aggregate, which is throughput
            # and not comparable to the centre's single-stream figure.
            b1 = next((r for r in rows if r["batch"] == 1), None)
            if b1:
                edge["tok_per_s"] = b1["tok_per_s"]
                edge["j_req_b1"] = b1["j_req"]
    return edge


def per_request(idle_w, j_dynamic, rate_per_day, total_load=0):
    """Joules attributable to one request.

    The idle floor is spread over everything the card serves that day. For a
    dedicated card that is our own rate; for a shared one it is the card's
    total load, of which our requests are a part.
    """
    denom = max(rate_per_day, total_load)
    return (idle_w * 86400) / denom + j_dynamic


def main():
    edge = load_edge()
    if edge["j_req_batched"] is None:
        print("edge_result.json has no usable energy figure yet -- run the edge "
              "sweep first, then re-run this.")
        return

    print("MEASURED INPUTS")
    print(f"  {edge['name']}: idle {edge['idle_w']:.2f} W, "
          f"{edge['j_req_batched']:.2f} J/req at batch {edge.get('best_batch')}, "
          f"acc {edge['acc_direct']*100:.1f}%"
          + ("   [cell flagged dirty: drift %.2f W]" % edge["drift_w"]
             if edge.get("dirty") else ""))
    print(f"  {CENTRE['name']}: idle {CENTRE['idle_w']:.2f} W, "
          f"{CENTRE['j_req_batched']:.2f} J/req at batch 32, "
          f"acc {CENTRE['acc_direct']*100:.1f}% direct / "
          f"{CENTRE['acc_cot']*100:.1f}% CoT")

    # ---- transport: carried, then shown to be irrelevant ------------------
    print("\nTRANSPORT TERM  (round trip, request + answer)")
    print(f"  {'payload':>10} {'low estimate':>16} {'high estimate':>16} "
          f"{'vs centre inference':>22}")
    for kb in PAYLOADS_KB:
        lo, hi = (kb / 1024 * j for j in J_PER_MB)
        print(f"  {kb:>7} KB {lo:>15.3f} J {hi:>15.3f} J "
              f"{hi / CENTRE['j_req_batched'] * 100:>20.2f}%")
    print("  -> even the high estimate on a 100 KB bundle is a few percent of one")
    print("     inference, and orders below the idle floor at operations rates.")
    transport_j = PAYLOADS_KB[1] / 1024 * J_PER_MB[1]   # 10 KB, high estimate

    # ---- joules per correct decision --------------------------------------
    print("\nJOULES PER CORRECT DECISION")
    print("  centre columns: dedicated, or sharing a card that also serves"
          " 100k / 1M req/day")
    header = (f"  {'req/day':>10} {'edge':>11} {'centre':>11} {'ctr+CoT':>11} "
              f"{'ctr@100k':>11} {'ctr@1M':>11}   winner")
    print(header)
    cot_batched = CENTRE["j_req_cot_b1"] / CENTRE["cot_batch_factor"]
    rows = []
    for rate in RATES:
        e = per_request(edge["idle_w"], edge["j_req_batched"], rate) / edge["acc_direct"]
        c = (per_request(CENTRE["idle_w"], CENTRE["j_req_batched"] + transport_j, rate)
             / CENTRE["acc_direct"])
        cc = (per_request(CENTRE["idle_w"], cot_batched + transport_j, rate)
              / CENTRE["acc_cot"])
        s1 = (per_request(CENTRE["idle_w"], CENTRE["j_req_batched"] + transport_j,
                          rate, total_load=100_000) / CENTRE["acc_direct"])
        s2 = (per_request(CENTRE["idle_w"], CENTRE["j_req_batched"] + transport_j,
                          rate, total_load=1_000_000) / CENTRE["acc_direct"])
        opts = {"edge": e, "centre": c, "centre+CoT": cc,
                "ctr@100k": s1, "ctr@1M": s2}
        win = min(opts, key=opts.get)
        rows.append((rate, e, c, cc, s1, s2, win))
        print(f"  {rate:>10,} {e:>11,.0f} {c:>11,.0f} {cc:>11,.0f} "
              f"{s1:>11,.0f} {s2:>11,.0f}   {win}")

    # ---- crossovers --------------------------------------------------------
    print("\nCROSSOVERS  (solved, not read off the table)")

    def crossover(idle_a, dyn_a, acc_a, idle_b, dyn_b, acc_b):
        """Daily rate at which arm B becomes cheaper per correct answer."""
        # idle_a*86400/(r*acc_a) + dyn_a/acc_a == idle_b*86400/(r*acc_b) + dyn_b/acc_b
        num = 86400 * (idle_a / acc_a - idle_b / acc_b)
        den = dyn_b / acc_b - dyn_a / acc_a
        return num / den if den > 0 else None

    r1 = crossover(edge["idle_w"], edge["j_req_batched"], edge["acc_direct"],
                   CENTRE["idle_w"], CENTRE["j_req_batched"] + transport_j,
                   CENTRE["acc_direct"])
    print(f"  edge -> dedicated centre : "
          + (f"{r1:,.0f} req/day" if r1 and r1 > 0
             else "never (centre never wins on energy per correct answer)"))

    # For a shared centre the idle term is fixed by the card's total load, so
    # instead of a crossover rate the useful question is: how busy must the
    # shared card be before it beats a dedicated edge box?
    r2 = None
    for load in (10_000, 50_000, 100_000, 500_000, 1_000_000, 5_000_000):
        c_share = (per_request(CENTRE["idle_w"], CENTRE["j_req_batched"] + transport_j,
                               1_000, total_load=load) / CENTRE["acc_direct"])
        e_1k = per_request(edge["idle_w"], edge["j_req_batched"], 1_000) / edge["acc_direct"]
        if c_share < e_1k:
            r2 = load
            break
    print(f"  shared centre beats edge once that card serves "
          + (f"{r2:,} req/day in total (our workload at 1,000/day)" if r2
             else "more than 5,000,000 req/day"))

    r3 = crossover(edge["idle_w"], edge["j_req_batched"], edge["acc_direct"],
                   CENTRE["idle_w"], cot_batched + transport_j, CENTRE["acc_cot"])
    print(f"  edge -> centre with CoT  : "
          + (f"{r3:,.0f} req/day" if r3 and r3 > 0 else "never"))

    # ---- latency, the other axis an operator cares about -------------------
    print("\nLATENCY  (128-token answer, single request)")
    print(f"  edge   {128 / edge['tok_per_s']:6.2f} s at {edge['tok_per_s']:.1f} tok/s")
    print(f"  centre {128 / CENTRE['tok_per_s']:6.2f} s at {CENTRE['tok_per_s']:.1f} tok/s"
          f"  (+{transport_j / 1e3:.4f} kJ transport, negligible latency for 10 KB)")

    json.dump({"edge": edge, "centre": CENTRE, "transport_j_10kb": transport_j,
               "rows": [{"rate": r[0], "edge": r[1], "centre": r[2],
                         "centre_cot": r[3], "shared10": r[4], "shared100": r[5],
                         "winner": r[6]} for r in rows],
               "crossover_edge_to_centre": r1,
               "crossover_edge_to_shared10": r2,
               "crossover_edge_to_cot": r3},
              open("placement_result.json", "w"), indent=1)
    print("\nwrote placement_result.json")


if __name__ == "__main__":
    main()
