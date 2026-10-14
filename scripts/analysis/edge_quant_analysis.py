"""What quantization does to the placement result.

The placement analysis compares one edge configuration against one datacenter
one, and its ranking rests on a single accuracy gap: 39.2% at the edge against
50.0% at the center. The paper says in as many words that a task where the
larger model's advantage is much wider could flip it. Four bits per weight
turns that caveat into a measurement, because it changes which model fits on
the edge card.

Three things come out, in the order they matter:

1. What quantization costs at fixed model size, which is not what it is
   usually assumed to cost.
2. What the memory it frees buys, which is where the accuracy gap goes.
3. Whether the placement ranking survives, recomputed on the same cost model
   the paper uses.

    python scripts/analysis/edge_quant_analysis.py [edge_quant_result.json]
"""
import json
import sys

# The published placement model, from Section 4.7 and Table 10.
CENTER = {
    "label": "L40S 14B bf16, batch 32",
    "acc": 0.500,
    "dyn_j": 55.95,
    "transport_j": 2.06,      # the 10 KB high-estimate round trip
    "idle_w": 86.93,
}
PUB_EDGE = {
    "label": "A2000 1.5B bf16, batch 32 (published)",
    "acc": 0.392,
    "dyn_j": 17.73,
    "transport_j": 0.0,
    "idle_w": 12.24,
}
VOLUMES = [100, 1_000, 10_000, 100_000, 1_000_000]


def j_per_correct(cfg, volume):
    idle = cfg["idle_w"] * 86400.0 / volume
    return (idle + cfg["dyn_j"] + cfg.get("transport_j", 0.0)) / cfg["acc"]


def main(path):
    d = json.load(open(path))
    rows = [e for e in d["results"] if "acc" in e]
    print("%s, batch %d, %d tasks, cold idle %.2f W\n"
          % (d["card"], d["batch"], len(rows[0]["rows"]), d["cold_idle"]))

    print("%-12s %7s %9s %9s %11s %8s %8s"
          % ("config", "VRAM", "accuracy", "J/req", "J/correct", "idle W",
             "req/s"))
    for e in rows:
        print("%-12s %6.2fG %8.1f%% %9.2f %11.1f %8.2f %8.2f"
              % ("%gB %s" % (e["params_b"], e["precision"]), e["vram_gib"],
                 e["acc"] * 100, e["energy"]["j_per_req"], e["j_per_correct"],
                 e["resident_idle_w"], e["energy"]["req_per_s"]))

    by = {("%g" % e["params_b"], e["precision"]): e for e in rows}

    # ---- 1: what four bits costs at fixed size -------------------------
    a, b = by.get(("1.5", "bf16")), by.get(("1.5", "nf4"))
    if a and b:
        print("\nquantization at fixed model size, 1.5B bf16 against nf4")
        print("   accuracy   %5.1f%% -> %5.1f%%   %+.1f points"
              % (a["acc"] * 100, b["acc"] * 100,
                 (b["acc"] - a["acc"]) * 100))
        print("   energy     %6.2f -> %6.2f J/req   %+.1f%%"
              % (a["energy"]["j_per_req"], b["energy"]["j_per_req"],
                 (b["energy"]["j_per_req"] / a["energy"]["j_per_req"] - 1) * 100))
        print("   memory     %6.2f -> %6.2f GiB   %.1fx less"
              % (a["vram_gib"], b["vram_gib"], a["vram_gib"] / b["vram_gib"]))
        print("   per correct decision %.1f -> %.1f J, %.2fx worse"
              % (a["j_per_correct"], b["j_per_correct"],
                 b["j_per_correct"] / a["j_per_correct"]))
        print("   Four bits is not free in either currency here: it loses")
        print("   accuracy and costs energy, because the weights are")
        print("   dequantized on the way through.")

    # ---- 2: what the freed memory buys ---------------------------------
    print("\nwhat the freed memory buys, against 1.5B bf16")
    if a:
        for key in (("3", "nf4"), ("7", "nf4")):
            e = by.get(key)
            if not e:
                continue
            print("   %-9s %+5.1f accuracy points, %5.2fx the energy, "
                  "%.2fx the joules per correct"
                  % ("%sB nf4" % key[0], (e["acc"] - a["acc"]) * 100,
                     e["energy"]["j_per_req"] / a["energy"]["j_per_req"],
                     e["j_per_correct"] / a["j_per_correct"]))
        best = min(rows, key=lambda e: e["j_per_correct"])
        print("   Lowest joules per correct decision: %gB %s at %.1f J"
              % (best["params_b"], best["precision"], best["j_per_correct"]))

    # ---- 3: does the placement ranking survive -------------------------
    print("\nplacement, joules per correct decision, same cost model as "
          "Table 10")
    arms = [PUB_EDGE]
    for e in rows:
        arms.append({
            "label": "A2000 %gB %s, batch %d (measured here)"
                     % (e["params_b"], e["precision"], d["batch"]),
            "acc": e["acc"], "dyn_j": e["energy"]["j_per_req"],
            "transport_j": 0.0, "idle_w": e["resident_idle_w"],
        })
    arms.append(CENTER)

    print("%-44s %s" % ("", "  ".join("%11s" % ("%d/day" % v)
                                      for v in VOLUMES)))
    for cfg in arms:
        print("%-44s %s" % (cfg["label"], "  ".join(
            "%11s" % "{:,.0f}".format(j_per_correct(cfg, v))
            for v in VOLUMES)))

    print("\n   Two caveats on reading the rows against each other. The")
    print("   published edge row carries a 12.24 W resident floor measured in")
    print("   another session; this card idles at %.2f W cold today, so the"
          % d["cold_idle"])
    print("   rows measured here sit on a cooler baseline. And they are taken")
    print("   at batch %d rather than 32, because that is what the 7B model"
          % d["batch"])
    print("   leaves room for.")

    # which arm wins at each volume
    print("\n   cheapest arm per correct decision, by volume")
    for v in VOLUMES:
        w = min(arms, key=lambda c: j_per_correct(c, v))
        print("      %9s/day   %s" % ("{:,}".format(v), w["label"]))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "edge_quant_result.json")
