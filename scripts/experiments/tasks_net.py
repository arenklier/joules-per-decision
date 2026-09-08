"""Verifiable network-operations tasks with generated ground truth.

Every task is produced by a generator that also computes the answer in plain
Python, so grading is exact-match against a value no model was involved in
deriving. Three families, each deterministic and each with a classical solver
that costs microseconds -- the same solvers are the millijoule baseline the
LLM arms get compared against later.
"""
import ipaddress
import random

# ---------------------------------------------------------------- subnetting


def gen_subnet(rng):
    base_prefix = rng.choice([16, 17, 18])
    new_prefix = base_prefix + rng.choice([3, 4, 5, 6])
    octet2 = rng.randrange(0, 256, 2 ** (24 - base_prefix) // 256 or 1)
    net = ipaddress.ip_network(f"10.{octet2}.0.0/{base_prefix}", strict=False)
    subs = list(net.subnets(new_prefix=new_prefix))
    idx = rng.randrange(1, min(len(subs), 32) + 1)
    field = rng.choice(["network", "broadcast", "last_usable"])
    sub = subs[idx - 1]
    if field == "network":
        gt, ask = str(sub.network_address), "network address"
    elif field == "broadcast":
        gt, ask = str(sub.broadcast_address), "broadcast address"
    else:
        gt = str(ipaddress.ip_address(int(sub.broadcast_address) - 1))
        ask = "last usable host address"
    prompt = (f"Subnet {net.with_prefixlen} into /{new_prefix} networks, "
              f"numbered from 1 in ascending order. "
              f"Give the {ask} of subnet number {idx}.")
    return prompt, gt, "ipv4 address"


# ----------------------------------------------------------------------- ACL

PROTOS = ["tcp", "udp"]


def gen_acl(rng):
    n = rng.randint(3, 5)
    aces, lines = [], []
    for i in range(n):
        action = rng.choice(["permit", "deny"])
        proto = rng.choice(PROTOS)
        net = ipaddress.ip_network(
            f"192.168.{rng.randrange(0, 8)}.0/{rng.choice([24, 25, 26])}")
        port = rng.choice([22, 80, 443, 3389, 8080])
        aces.append((action, proto, net, port))
        lines.append(f"{i+1}. {action} {proto} {net.with_prefixlen} "
                     f"any eq {port}")
    # a packet that hits one of the rules often enough to be interesting
    hit = rng.random() < 0.75
    if hit:
        a = rng.choice(aces)
        src = ipaddress.ip_address(int(a[2].network_address) + rng.randrange(
            1, max(2, a[2].num_addresses - 1)))
        proto, port = a[1], a[3]
    else:
        src = ipaddress.ip_address(f"172.16.{rng.randrange(0,255)}.5")
        proto, port = rng.choice(PROTOS), rng.choice([53, 123, 161])

    verdict = "DENY"                       # implicit deny at the end
    for action, p, net, prt in aces:       # first match wins
        if p == proto and src in net and port == prt:
            verdict = action.upper()
            break

    prompt = ("Evaluate this Cisco-style ACL, first match wins, with an "
              "implicit deny at the end:\n" + "\n".join(lines) +
              f"\n\nPacket: proto={proto} src={src} dst=any dport={port}. "
              "Is it PERMIT or DENY?")
    return prompt, verdict, "PERMIT or DENY"


# ------------------------------------------------------------ BGP best path

ORIGIN_RANK = {"IGP": 0, "EGP": 1, "INCOMPLETE": 2}


def gen_bgp(rng):
    n = rng.randint(3, 4)
    routes, lines = [], []
    # keep weight and local_pref tied often enough that later tiebreakers matter
    shared_weight = rng.choice([0, 0, 100])
    shared_lp = rng.choice([100, 100, 150])
    for i in range(n):
        r = {
            "id": f"R{i+1}",
            "weight": shared_weight if rng.random() < 0.7 else rng.choice([0, 50, 100]),
            "local_pref": shared_lp if rng.random() < 0.7 else rng.choice([100, 150, 200]),
            "as_path": rng.randint(1, 4),
            "origin": rng.choice(["IGP", "IGP", "EGP", "INCOMPLETE"]),
            "med": rng.choice([0, 10, 50, 100]),
        }
        routes.append(r)
        lines.append(f"{r['id']}: weight={r['weight']} local_pref={r['local_pref']} "
                     f"as_path_length={r['as_path']} origin={r['origin']} "
                     f"med={r['med']}")

    best = sorted(routes, key=lambda r: (-r["weight"], -r["local_pref"],
                                         r["as_path"], ORIGIN_RANK[r["origin"]],
                                         r["med"], r["id"]))[0]
    # only keep unambiguous instances
    key = lambda r: (-r["weight"], -r["local_pref"], r["as_path"],
                     ORIGIN_RANK[r["origin"]], r["med"])
    if sum(1 for r in routes if key(r) == key(best)) > 1:
        return None

    prompt = ("Apply the BGP best-path selection algorithm to these routes to "
              "the same prefix, in the standard order (highest weight, then "
              "highest local preference, then shortest AS path, then lowest "
              "origin type with IGP<EGP<INCOMPLETE, then lowest MED):\n"
              + "\n".join(lines) + "\n\nWhich route is selected?")
    return prompt, best["id"], "route id such as R2"


GENERATORS = [("subnet", gen_subnet), ("acl", gen_acl), ("bgp", gen_bgp)]


def build(n_per_family=16, seed=20260903):
    rng = random.Random(seed)
    tasks = []
    for fam, gen in GENERATORS:
        made = 0
        guard = 0
        while made < n_per_family and guard < 2000:
            guard += 1
            out = gen(rng)
            if out is None:
                continue
            prompt, gt, fmt = out
            tasks.append({"family": fam, "id": f"{fam}_{made:02d}",
                          "prompt": prompt, "gt": gt, "fmt": fmt})
            made += 1
    return tasks


def grade(answer_text, gt):
    """Exact match on the value after the last ANSWER: marker."""
    if not answer_text:
        return False, ""
    upper = answer_text.upper()
    idx = upper.rfind("ANSWER:")
    tail = answer_text[idx + 7:] if idx >= 0 else answer_text
    got = tail.strip().splitlines()[0].strip() if tail.strip() else ""
    got = got.strip().strip("`*.,;:'\"").strip()
    return got.upper() == gt.upper(), got


if __name__ == "__main__":
    ts = build()
    print(f"{len(ts)} tasks")
    for t in ts[:2] + ts[8:10] + ts[16:18]:
        print(f"\n--- {t['id']} (gt={t['gt']}) ---\n{t['prompt']}")
