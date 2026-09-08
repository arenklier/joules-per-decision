"""Grade NetArena's own reference solution against itself.

The paper claims this check passed 5/5 before any model output was scored.
It is not in any script or log here, so it is re-run: feed each task's
ground-truth process_graph back through the same harness that grades model
output. A reference solution that fails its own grader would mean the
grading path, not the models, produces the accuracy figures.

No GPU and no model involved -- this only executes the reference code.
"""
import json
import os
import re
import sys

os.environ.setdefault("HF_HOME", "/mnt/data/hf")
sys.path.insert(0, "/mnt/data/joule-spike/NetArena/app-malt")

from solid_step_helper import getGraphData
from malt_env import BenchmarkEvaluator

N = int(os.environ.get("N", 5))
SAMPLE = os.environ.get("SAMPLE", "/mnt/data/joule-spike/netarena_sample.json")

print("loading MALT graph ...", flush=True)
_, G = getGraphData()
ev = BenchmarkEvaluator(G)
print(f"graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges", flush=True)

def as_submission(code):
    """What NetArena's own extractor does to a candidate answer: any function
    whose name contains process_graph becomes process_graph, imports dropped."""
    code = re.sub(r"def\s+([a-zA-Z_0-9]*process_graph[a-zA-Z_0-9]*)",
                  "def process_graph", code)
    start = code.find("def process_graph")
    code = code[start:] if start >= 0 else code
    return "\n".join(l for l in code.split("\n")
                     if not l.strip().startswith("import"))


tasks = json.load(open(SAMPLE, encoding="utf-8"))[:N]
ok = 0
for i, t in enumerate(tasks, 1):
    gt = as_submission(t["gt_code"])
    try:
        (ret, gt_ret, v_ok, v_err, gv_ok, gv_err, ret_graph) = ev.run_agent_output(
            t["question"], t["gt_code"], gt)
        res = ev.ground_truth_check(t["question"], t["label"], ret, gt_ret,
                                    ret_graph, v_ok, v_err, gv_ok, gv_err, 0.0)
        passed = res.get("Result-Correctness") == "Pass"
    except Exception as e:
        passed, res = False, {"Error": repr(e)[:120]}
    ok += passed
    print(f"  [{i}/{len(tasks)}] {t['id']:>22}  label={t['label']:<12} "
          f"{'PASS' if passed else 'FAIL ' + str(res.get('Error'))[:70]}", flush=True)

print(f"\nreference solution graded against itself: {ok}/{len(tasks)}")
sys.exit(0 if ok == len(tasks) else 1)
