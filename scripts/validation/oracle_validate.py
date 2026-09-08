"""Independent re-derivation of the internal task set's ground truth.

The reviewers asked for the equivalent of NetArena's reference-solution
self-check (5/5) on this study's own Python oracle. tasks_net.py both
generates each task and computes its answer, so a bug there is invisible:
the answer would be wrong and the grader would agree with it.

This script re-derives every answer a second time, deliberately by a
different route than tasks_net uses, and reports disagreements:

  subnetting  tasks_net slices with ipaddress.ip_network.subnets(); here the
              subnet boundaries are recomputed from raw integer arithmetic on
              the address space, with no ipaddress subnetting involved.
  acl         tasks_net evaluates the rule list in generation order; here the
              packet is matched by re-parsing the rendered prompt text, so a
              mismatch between what the model was shown and what was scored
              would surface.
  bgp         tasks_net sorts on a tuple key; here best-path is selected by
              sequential pairwise elimination in the documented attribute
              order, which is how the algorithm is specified rather than how
              it is convenient to implement.

Agreement is not proof the task set is well designed. It only rules out the
specific failure where the generator and the grader share one bug.
"""
import ipaddress
import re
import sys

from tasks_net import build, GENERATORS


# ----------------------------------------------------------------- subnetting

def redo_subnet(prompt, claimed):
    """Recompute from integer arithmetic, no ipaddress.subnets()."""
    m = re.match(r"Subnet (\S+) into /(\d+) networks, numbered from 1 in "
                 r"ascending order\. Give the (.+?) of subnet number (\d+)\.",
                 prompt)
    if not m:
        return None, "prompt did not parse"
    base, newpfx, field, idx = m.group(1), int(m.group(2)), m.group(3), int(m.group(4))

    net = ipaddress.ip_network(base)
    base_int = int(net.network_address)
    block = 1 << (32 - newpfx)               # addresses per subnet
    start = base_int + (idx - 1) * block     # nth subnet, 1-indexed
    end = start + block - 1

    if field == "network address":
        got = start
    elif field == "broadcast address":
        got = end
    elif field == "last usable host address":
        got = end - 1
    else:
        return None, f"unknown field {field!r}"
    return str(ipaddress.ip_address(got)), None


# ------------------------------------------------------------------------ acl

def redo_acl(prompt, claimed):
    """Re-parse the rendered ACL text and match the packet against it."""
    rules = re.findall(
        r"^\s*\d+\.\s+(permit|deny)\s+(tcp|udp)\s+(\S+)\s+any\s+eq\s+(\d+)",
        prompt, re.MULTILINE)
    pm = re.search(r"Packet: proto=(\w+) src=(\S+) dst=any dport=(\d+)", prompt)
    if not rules or not pm:
        return None, "prompt did not parse"
    proto, src, dport = pm.group(1), ipaddress.ip_address(pm.group(2)), int(pm.group(3))

    for action, rproto, rnet, rport in rules:
        if rproto == proto and src in ipaddress.ip_network(rnet) and int(rport) == dport:
            return action.upper(), None
    return "DENY", None                       # implicit deny


# ------------------------------------------------------------------------ bgp

ORIGIN = {"IGP": 0, "EGP": 1, "INCOMPLETE": 2}


def redo_bgp(prompt, claimed):
    """Sequential pairwise elimination in the documented attribute order."""
    rows = re.findall(
        r"^(R\d+): weight=(\d+) local_pref=(\d+) as_path_length=(\d+) "
        r"origin=(\w+) med=(\d+)", prompt, re.MULTILINE)
    if not rows:
        return None, "prompt did not parse"
    routes = [{"id": r[0], "weight": int(r[1]), "lp": int(r[2]),
               "asp": int(r[3]), "org": ORIGIN[r[4]], "med": int(r[5])}
              for r in rows]

    # eliminate stage by stage, exactly as the algorithm is stated
    cand = routes
    for key, better in (("weight", max), ("lp", max), ("asp", min),
                        ("org", min), ("med", min)):
        target = better(r[key] for r in cand)
        cand = [r for r in cand if r[key] == target]
        if len(cand) == 1:
            return cand[0]["id"], None
    return None, f"ambiguous after all tiebreakers: {[r['id'] for r in cand]}"


REDO = {"subnet": redo_subnet, "acl": redo_acl, "bgp": redo_bgp}


def main():
    n_per = int(sys.argv[1]) if len(sys.argv) > 1 else 40
    tasks = build(n_per_family=n_per)
    print(f"re-deriving {len(tasks)} tasks ({n_per} per family)\n")

    totals, agree, problems = {}, {}, []
    for t in tasks:
        fam = t["family"]
        totals[fam] = totals.get(fam, 0) + 1
        got, err = REDO[fam](t["prompt"], t["gt"])
        if err:
            problems.append((t["id"], t["gt"], None, err))
        elif got != t["gt"]:
            problems.append((t["id"], t["gt"], got, "MISMATCH"))
        else:
            agree[fam] = agree.get(fam, 0) + 1

    for fam, _ in GENERATORS:
        a, n = agree.get(fam, 0), totals.get(fam, 0)
        print(f"  {fam:8s} {a}/{n} agree" + ("" if a == n else "   <-- CHECK"))

    total_a, total_n = sum(agree.values()), sum(totals.values())
    print(f"\n  overall  {total_a}/{total_n} agree")

    if problems:
        print("\ndisagreements:")
        for tid, gt, got, why in problems[:20]:
            print(f"  {tid}: oracle={gt!r} independent={got!r}  ({why})")
    return 0 if not problems else 1


if __name__ == "__main__":
    sys.exit(main())
