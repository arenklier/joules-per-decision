"""The classical arm: deterministic solvers on the same 48 prompts.

Fairness rule: the solvers get exactly what the model got -- the natural
language prompt -- and must parse it themselves. Parsing is part of the cost,
not a freebie. Nothing here reads the ground truth; correctness is checked
against it afterwards, the same way the model's answers were.

Energy is deliberately NOT claimed as a measurement. RAPL is not exposed inside
this VM and perf_event_paranoid blocks the platform counter, so the honest
statement is that a solver's cost sits below what the instrument can resolve.
What is measured is CPU time; the joule figure is a stated bound derived from
per-core power, and the comparison survives that bound being wrong by 10x.
"""
import ipaddress
import re
import statistics as st
import time

from tasks_net import build

# Xeon Gold 6430: 32 cores, 270 W TDP -> ~8.4 W per fully loaded core. Used only
# to bound the classical arm, never to state a measured value.
W_PER_CORE = 8.4

ORIGIN_RANK = {"IGP": 0, "EGP": 1, "INCOMPLETE": 2}


def solve_subnet(prompt):
    m = re.search(r"Subnet (\S+) into /(\d+) networks", prompt)
    net = ipaddress.ip_network(m.group(1))
    new_prefix = int(m.group(2))
    idx = int(re.search(r"subnet number (\d+)", prompt).group(1))
    sub = next(s for i, s in enumerate(net.subnets(new_prefix=new_prefix), 1)
               if i == idx)
    if "network address" in prompt:
        return str(sub.network_address)
    if "broadcast address" in prompt:
        return str(sub.broadcast_address)
    return str(ipaddress.ip_address(int(sub.broadcast_address) - 1))


ACE_RE = re.compile(r"^\d+\.\s+(permit|deny)\s+(tcp|udp)\s+(\S+)\s+any\s+eq\s+(\d+)",
                    re.M)
PKT_RE = re.compile(r"Packet: proto=(\w+) src=(\S+) dst=any dport=(\d+)")


def solve_acl(prompt):
    pkt = PKT_RE.search(prompt)
    proto, src, port = pkt.group(1), ipaddress.ip_address(pkt.group(2)), int(pkt.group(3))
    for action, p, cidr, prt in ACE_RE.findall(prompt):
        if p == proto and src in ipaddress.ip_network(cidr) and int(prt) == port:
            return action.upper()
    return "DENY"


ROUTE_RE = re.compile(
    r"^(R\d+): weight=(\d+) local_pref=(\d+) as_path_length=(\d+) "
    r"origin=(\w+) med=(\d+)", re.M)


def solve_bgp(prompt):
    routes = [(rid, int(w), int(lp), int(ap), org, int(med))
              for rid, w, lp, ap, org, med in ROUTE_RE.findall(prompt)]
    return min(routes, key=lambda r: (-r[1], -r[2], r[3], ORIGIN_RANK[r[4]], r[5]))[0]


SOLVERS = {"subnet": solve_subnet, "acl": solve_acl, "bgp": solve_bgp}


def main():
    tasks = build()
    print(f"{len(tasks)} tasks, classical arm\n")

    wrong = []
    for t in tasks:
        got = SOLVERS[t["family"]](t["prompt"])
        if got.upper() != t["gt"].upper():
            wrong.append((t["id"], t["gt"], got))
    acc = 1 - len(wrong) / len(tasks)
    print(f"accuracy: {acc*100:.1f}%  ({len(tasks)-len(wrong)}/{len(tasks)})")
    for w in wrong:
        print("   MISMATCH", w)

    # amortise: a single solve is far too short to time, exactly as a single
    # direct generation was too short to integrate
    per_family = {}
    for fam in ("subnet", "acl", "bgp"):
        sub = [t for t in tasks if t["family"] == fam]
        fn = SOLVERS[fam]
        reps, elapsed = 200, 0.0
        while elapsed < 2.0:                      # grow until the window is real
            reps *= 2
            t0 = time.perf_counter()
            for _ in range(reps):
                for t in sub:
                    fn(t["prompt"])
            elapsed = time.perf_counter() - t0
            if reps > 200000:
                break
        per_call = elapsed / (reps * len(sub))
        per_family[fam] = per_call
        print(f"{fam:8s} {per_call*1e6:8.2f} us/task   "
              f"({reps} reps x {len(sub)} tasks in {elapsed:.2f} s)")

    mean_s = st.mean(per_family.values())
    j_bound = mean_s * W_PER_CORE
    print(f"\nmean {mean_s*1e6:.2f} us/task")
    print(f"bounded energy at {W_PER_CORE} W/core: {j_bound*1e6:.1f} uJ/task "
          f"({j_bound:.3e} J)")

    print("\nversus the measured LLM arms (J per task, same 48 tasks):")
    for name, j, a in (("Qwen2.5-1.5B direct", 9.01, 0.438),
                       ("Qwen2.5-3B  direct", 19.13, 0.438),
                       ("Qwen2.5-7B  direct", 40.08, 0.500),
                       ("Qwen2.5-14B direct", 79.85, 0.521),
                       ("Qwen2.5-14B CoT", 3814.8, 0.667)):
        print(f"  {name:22s} {j:8.2f} J  acc {a*100:4.1f}%   "
              f"ratio {j/j_bound:.2e}x   per correct answer "
              f"{(j/a)/(j_bound/acc):.2e}x")


if __name__ == "__main__":
    main()
