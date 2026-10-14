"""Check the agent loop's control flow without spending GPU time.

Three things cannot be fixed once a run is on the card, so they are settled
here first: that the loop extracts answers exactly as the grader does, that
repair terminates instead of spinning, and above all that the check stays
blind to an answer which is well formed and merely wrong. That last property
is what separates a measurement from a rigged win, and it is asserted rather
than asserted in prose.

Run from anywhere:  python scripts/validation/agent_loop_selfcheck.py
"""
import contextlib
import os
import sys
import types

HERE = os.path.dirname(os.path.abspath(__file__))
# Repo layout puts the experiment next door; the measurement host keeps every
# script in one flat directory, so fall back to this file's own.
EXP = os.path.join(os.path.dirname(HERE), "experiments")
sys.path.insert(0, EXP if os.path.isdir(EXP) else HERE)

# agent_loop pulls in torch, transformers and the sampler at import time.
# Stub them so the pure-python control flow can be exercised on any machine.
for name in ("torch", "transformers", "power_sampler"):
    sys.modules.setdefault(name, types.ModuleType(name))
sys.modules["torch"].inference_mode = contextlib.nullcontext
sys.modules["torch"].cuda = types.SimpleNamespace(
    synchronize=lambda: None, empty_cache=lambda: None)
sys.modules["torch"].bfloat16 = None
sys.modules["transformers"].AutoTokenizer = object
sys.modules["transformers"].AutoModelForCausalLM = object
sys.modules["power_sampler"].PowerSampler = object

import agent_loop as al                                   # noqa: E402
from tasks_net import build                               # noqa: E402

TASKS = build(n_per_family=al.N_PER_FAMILY)
for _i, _t in enumerate(TASKS):          # the loop reads t["tpl"]
    _t["tpl"] = _i % 3
BY_ID = {t["id"]: t for t in TASKS}


class ScriptedModel:
    """Canned answers, one list per task, consumed in call order."""

    def __init__(self, plan):
        self.plan = plan
        self.seen = {}

    def __call__(self, model, tok, prompt, max_new):
        tid = next(t["id"] for t in TASKS if t["prompt"] in prompt)
        i = self.seen.get(tid, 0)
        self.seen[tid] = i + 1
        seq = self.plan.get(tid, ["junk"])
        return "ANSWER: " + (seq[i] if i < len(seq) else seq[-1]), 9


def trace(task, answers, reflect, repair):
    al.generate = ScriptedModel({task["id"]: answers})
    return al.run_task(None, None, task, reflect, repair)


def main():
    sub, acl = BY_ID["subnet_00"], BY_ID["acl_00"]

    assert al.extract("ANSWER: 10.4.0.0") == "10.4.0.0"
    assert al.extract("blah\nANSWER: R2\n") == "R2"
    assert al.extract("ANSWER: `PERMIT`.") == "PERMIT"
    assert al.extract("reasoning\nANSWER: 1\nANSWER: 2") == "2"
    print("extractor is the grader's own rule")

    r = trace(sub, ["junk"], False, False)
    assert r["calls"] == 1 and r["checks_failed"] == 0, r
    print("single           1 call, no check")

    r = trace(sub, ["not-an-address", sub["gt"]], False, True)
    assert r["calls"] == 2 and r["checks_failed"] == 1, r
    assert r["answers_changed"] == 1 and r["correct"], r
    print("repair           fires once, fixes, terminates")

    r = trace(sub, ["bad"] * 9, False, True)
    assert r["calls"] == 1 + al.MAX_REPAIRS, r
    assert not r["valid"] and not r["correct"], r
    print("repair           bounded at k=%d, no spin" % al.MAX_REPAIRS)

    # The property the whole experiment rests on.
    flipped = "DENY" if acl["gt"] == "PERMIT" else "PERMIT"
    r = trace(acl, [flipped], False, True)
    assert r["calls"] == 1 and r["checks_failed"] == 0, r
    assert r["valid"] and not r["correct"], r
    print("repair           blind to a well-formed wrong answer")

    r = trace(sub, ["junk", sub["gt"]], True, False)
    assert r["calls"] == 2, r
    print("reflect          exactly 2 calls, unconditionally")

    r = trace(sub, ["junk", "still-bad", sub["gt"]], True, True)
    assert r["calls"] == 3 and r["checks_failed"] == 1 and r["correct"], r
    print("reflect+repair   2 + repairs, composes")

    print("\nall control-flow checks pass")


if __name__ == "__main__":
    main()
