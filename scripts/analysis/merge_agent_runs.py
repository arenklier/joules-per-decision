"""Assemble the agent-loop result from the sessions that produced it.

The first session covered all four model sizes, but a foreign job entered the
GPU while 7B was being measured. The contention guard caught it: the reflect
cell closed 247 W above its opening baseline and read 174.50 J per task, and
the reflect+repair cell returned -11.75 J, a negative energy, because its own
opening baseline had been taken while the intruder was resident. Those cells
are discarded rather than corrected -- this is the paper's first pitfall
happening to the paper -- and 7B was re-measured alone in a clean session.

The two sessions agree where both are clean, which is what licenses splicing
them: 7B single read 45.02 J against 44.52, and 7B repair 62.63 against 62.76,
a 1.1% and 0.2% difference across independent sessions.

Every discarded and substituted cell is recorded in the output under
`provenance`, so the deposit carries the decision and not just its result.

    python scripts/analysis/merge_agent_runs.py \\
        agent_loop_result_run1.json agent_loop_7b_result.json \\
        -o agent_loop_result.json
"""
import argparse
import json


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("base", help="the multi-model session")
    ap.add_argument("override", nargs="+",
                    help="later sessions whose cells replace the base's")
    ap.add_argument("-o", "--out", default="agent_loop_result.json")
    args = ap.parse_args()

    base = json.load(open(args.base))
    by_params = {e["params_b"]: e for e in base["results"]}
    prov = []

    for path in args.override:
        d = json.load(open(path))
        for e in d["results"]:
            pb = e["params_b"]
            old = by_params.get(pb)
            if old is not None:
                dirty = [a for a in ("single", "repair", "reflect",
                                     "reflect+repair")
                         if isinstance(old.get(a), dict) and old[a].get("dirty")]
                prov.append({
                    "params_b": pb, "replaced_from": args.base,
                    "replaced_with": path,
                    "discarded_cells_flagged_dirty": dirty,
                    "reason": "foreign process on the GPU during the first "
                              "session; contention guard flagged the cells and "
                              "one returned negative net energy",
                })
            by_params[pb] = e

    out = dict(base)
    out["results"] = [by_params[k] for k in sorted(by_params)]
    out["provenance"] = prov
    json.dump(out, open(args.out, "w"), indent=1)

    print("wrote %s" % args.out)
    for e in out["results"]:
        cells = [a for a in ("single", "repair", "reflect", "reflect+repair")
                 if isinstance(e.get(a), dict)]
        bad = [a for a in cells if e[a].get("dirty")]
        print("  %-6s %d cells%s" % ("%gB" % e["params_b"], len(cells),
                                     "   still dirty: " + ", ".join(bad)
                                     if bad else ""))
    for p in prov:
        print("\n  %gB replaced: %s -> %s" % (p["params_b"],
                                              p["replaced_from"],
                                              p["replaced_with"]))
        print("    discarded (flagged dirty): %s"
              % (", ".join(p["discarded_cells_flagged_dirty"]) or "none"))


if __name__ == "__main__":
    main()
