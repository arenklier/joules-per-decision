"""Ground-truth-free output checks for the agent loop.

A repair loop is only honest if its validator does not know the answer. If the
checker were the grader, the loop would be sitting an exam with the answer key
on the desk: enough iterations and accuracy goes to 100% by construction, and
the measurement means nothing.

What a real network deployment has instead is a linter or a commit-check: it
can tell you a config is malformed, out of range, or names a rule that does not
exist, without knowing which config you meant. That is what these are. Every
check is derived from the prompt alone.

The contract is enforced by the signature. No function here takes a ground
truth, and `selftest()` asserts that a correct answer is never rejected -- a
validator that fails good answers would make the loop actively harmful and
would confound the experiment it is built for.

Validator strength is deliberately uneven across the three families, and that
asymmetry is itself a result: a repair loop can only pay for itself where a
ground-truth-free check exists.

    subnet   strong   alignment and containment pin a large error class
    bgp      medium   the chosen route must be one of the offered ones
    acl      weak     only the verdict vocabulary can be checked

Anything stronger for ACL would have to evaluate the rule list, which is the
answer.
"""
import ipaddress
import re

FAMILIES = ("subnet", "acl", "bgp")


# --------------------------------------------------------------- subnetting

_SUBNET_RE = re.compile(
    r"Subnet\s+(?P<parent>\d+\.\d+\.\d+\.\d+/\d+)\s+into\s+/(?P<newpfx>\d+)")
_FIELD_RE = re.compile(
    r"Give the (?P<field>network address|broadcast address|"
    r"last usable host address)")


def _check_subnet(prompt, answer):
    m = _SUBNET_RE.search(prompt)
    f = _FIELD_RE.search(prompt)
    if not m or not f:
        return True, ""                      # unparseable prompt: do not judge
    parent = ipaddress.ip_network(m.group("parent"), strict=False)
    new_pfx = int(m.group("newpfx"))
    field = f.group("field")

    try:
        addr = ipaddress.ip_address(answer)
    except ValueError:
        return False, f"{answer!r} is not a valid IPv4 address."

    if addr not in parent:
        return False, (f"{addr} lies outside {parent.with_prefixlen}, the "
                       f"network being subnetted.")

    # Offset inside the parent, modulo one /new_pfx block. Note that the task
    # also states WHICH subnet is wanted; that index is deliberately not used,
    # because combining it with the alignment rule would determine the answer
    # outright and the check would be the grader in disguise.
    block = 1 << (32 - new_pfx)
    offset = (int(addr) - int(parent.network_address)) % block
    want = {"network address": 0,
            "broadcast address": block - 1,
            "last usable host address": block - 2}[field]
    if offset != want:
        where = {0: "the first address of its /%d block" % new_pfx,
                 block - 1: "the last address of its /%d block" % new_pfx,
                 block - 2: "the second-to-last address of its /%d block"
                            % new_pfx}[want]
        return False, (f"The {field} must be {where}; {addr} sits "
                       f"{offset} addresses into a block of {block}.")
    return True, ""


# ---------------------------------------------------------------------- ACL

def _check_acl(prompt, answer):
    verdict = answer.strip().upper()
    if verdict not in ("PERMIT", "DENY"):
        return False, (f"{answer!r} is not a verdict; answer exactly PERMIT "
                       f"or DENY.")
    return True, ""


# ---------------------------------------------------------------------- BGP

_ROUTE_DECL_RE = re.compile(r"^(R\d+):", re.M)


def _check_bgp(prompt, answer):
    offered = _ROUTE_DECL_RE.findall(prompt)
    if not offered:
        return True, ""
    got = answer.strip().upper()
    if not re.fullmatch(r"R\d+", got):
        return False, (f"{answer!r} is not a route identifier; answer with one "
                       f"of {', '.join(offered)}.")
    if got not in offered:
        return False, (f"Route {got} was not among the routes offered "
                       f"({', '.join(offered)}).")
    return True, ""


_CHECKS = {"subnet": _check_subnet, "acl": _check_acl, "bgp": _check_bgp}


def validate(family, prompt, answer):
    """Check an answer against the prompt alone.

    Returns (ok, complaint). The complaint is what the repair turn is shown;
    greedy decoding means an identical prompt reproduces an identical wrong
    answer, so the complaint is what makes repair possible at all.
    """
    if answer is None:
        return False, "No answer was produced."
    fn = _CHECKS.get(family)
    if fn is None:
        return True, ""
    return fn(prompt, str(answer))


# -------------------------------------------------------------------- tests

def selftest():
    """Two invariants, both of which must hold before any GPU time is spent.

    1. No correct answer is ever rejected. A validator that fails good answers
       turns the repair loop into a source of error rather than a fix.
    2. The checks actually bite. A validator that accepts everything collapses
       the loop into the single-shot arm and the experiment measures nothing.
    """
    import sys
    sys.path.insert(0, __file__.rsplit("/", 1)[0].rsplit("\\", 1)[0])
    from tasks_net import build

    tasks = build(n_per_family=40)
    print("%d tasks" % len(tasks))

    # 1 -- correct answers must always pass
    bad = []
    for t in tasks:
        ok, why = validate(t["family"], t["prompt"], t["gt"])
        if not ok:
            bad.append((t["id"], t["gt"], why))
    print("\ncorrect answers rejected: %d" % len(bad))
    for b in bad[:5]:
        print("   ", b)
    assert not bad, "validator rejects correct answers"

    # 2 -- how far the checks reach, by kind of error rather than in aggregate
    #
    # A single catch rate flatters the weak families, because most synthetic
    # wrong answers are malformed and malformation is the one thing every
    # check sees. What decides whether a repair loop can pay for itself is the
    # last column: answers that are well formed and simply wrong.
    kinds = ("malformed", "wrong_shape", "wrong_value")
    tally = {f: {k: [0, 0] for k in kinds} for f in FAMILIES}
    for t in tasks:
        fam = t["family"]
        for kind, wrong in _plausible_wrong(fam, t):
            ok, _ = validate(fam, t["prompt"], wrong)
            tally[fam][kind][1] += 1
            tally[fam][kind][0] += (not ok)

    print("\ncaught, by family and kind of error:")
    print("   %-8s %12s %12s %12s" % ("", *kinds))
    for f in FAMILIES:
        cells = []
        for k in kinds:
            c, n = tally[f][k]
            cells.append("-" if not n else "%3d/%-3d %4.0f%%"
                         % (c, n, 100.0 * c / n))
        print("   %-8s %12s %12s %12s" % (f, *cells))

    # Malformed output must always be caught: that is the floor.
    for f in FAMILIES:
        c, n = tally[f]["malformed"]
        assert c == n, "%s lets malformed output through" % f

    # Only subnetting can reach past shape into value, and it is bounded even
    # there: an answer landing on the adjacent block is correctly aligned and
    # passes. That ceiling is the honest limit of a ground-truth-free check,
    # and it is why the loop is a measurement rather than a rigged win.
    assert tally["subnet"]["wrong_shape"][0] == tally["subnet"]["wrong_shape"][1]
    assert tally["acl"]["wrong_value"][0] == 0, "acl check is seeing the answer"
    assert tally["bgp"]["wrong_value"][0] == 0, "bgp check is seeing the answer"
    print("\nok")


def _plausible_wrong(family, task):
    """Wrong answers of the kind a model actually produces, labelled by kind.

    malformed    not a usable value at all
    wrong_shape  a usable value in a structurally impossible position
    wrong_value  well formed, correctly shaped, and simply the wrong answer
    """
    gt = task["gt"]
    if family == "subnet":
        a = int(ipaddress.ip_address(gt))
        m = _SUBNET_RE.search(task["prompt"])
        block = 1 << (32 - int(m.group("newpfx"))) if m else 256
        return [("malformed", "not an address"),
                ("malformed", ""),
                ("wrong_shape", str(ipaddress.ip_address(a + 1))),
                ("wrong_shape", str(ipaddress.ip_address(a - 1))),
                # the adjacent block: off by one whole subnet, perfectly
                # aligned, and invisible to any check that does not know
                # which subnet was asked for
                ("wrong_value", str(ipaddress.ip_address(a + block)))]
    if family == "acl":
        return [("malformed", "allow"),
                ("malformed", "The packet is permitted."),
                ("malformed", ""),
                ("wrong_value", "PERMIT" if gt == "DENY" else "DENY")]
    offered = _ROUTE_DECL_RE.findall(task["prompt"])
    other = next((r for r in offered if r != gt), gt)
    return [("malformed", "the first one"),
            ("malformed", ""),
            ("wrong_shape", "R9"),
            ("wrong_value", other)]


if __name__ == "__main__":
    selftest()
