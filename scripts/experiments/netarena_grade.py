"""Accuracy pass on NetArena, using its own grading harness.

The energy run (netarena_energy.py) already measured joules and tokens on 60
sampled tasks across four models, base vs. chain-of-thought. It never kept the
generated text -- the logits were consumed for a token count and discarded --
so no accuracy number exists yet for this benchmark. Decoding here is greedy
(do_sample=False), so re-running generation reproduces the same text the
energy run saw; this is a second pass over the same deterministic outputs,
not a new experiment.

Grading itself is NetArena's own: BenchmarkEvaluator.run_agent_output execs
the extracted process_graph function against the real 1.9 MB MALT graph and
compares to ground_truth_process_graph via ground_truth_check's per-type
comparator (graph isomorphism for graph results, exact match otherwise).
Nothing about the comparison logic is written here; app-malt's own code
decides correct vs. wrong.
"""
import copy, gc, json, os, re, signal, sys, time, traceback

os.environ.setdefault("HF_HOME", "/mnt/data/hf")
os.environ.setdefault("TRANSFORMERS_CACHE", "/mnt/data/hf")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "NetArena", "app-malt"))

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from solid_step_helper import getGraphData
from malt_env import BenchmarkEvaluator

# text_utils.py imports `netarena.agent_client`, a package this run has no use
# for and does not install; extract_code_output itself has no such dependency,
# so it is copied verbatim here rather than patching the vendored repo file.
def extract_code_output(answer: str):
    regex = re.compile(r'def\s+([a-zA-Z_0-9]*process_graph[a-zA-Z_0-9]*)')
    answer = regex.sub('def process_graph', answer)
    start = answer.find("def process_graph")
    if start == -1:
        return ""
    code_block_end = answer.find("```", answer.find("```", start))
    if code_block_end != -1:
        clean_code = answer[start:code_block_end].strip()
    else:
        clean_code = answer[start:].strip()
    clean_code = '\n'.join(line for line in clean_code.split('\n')
                           if not line.strip().startswith("import"))
    return clean_code

_ALL_MODELS = [
    "Qwen/Qwen2.5-1.5B-Instruct",
    "Qwen/Qwen2.5-3B-Instruct",
    "Qwen/Qwen2.5-7B-Instruct",
    "Qwen/Qwen2.5-14B-Instruct",
]
_size_filter = os.environ.get("MODEL_SIZES")
MODELS = ([m for m in _ALL_MODELS if any(s in m for s in _size_filter.split(","))]
          if _size_filter else _ALL_MODELS)
BASE_MAX = int(os.environ.get("BASE_MAX", 768))
COT_MAX = int(os.environ.get("COT_MAX", 2048))
EXEC_TIMEOUT_S = 8
OUT = os.environ.get("OUT", "netarena_grade_result.json")
SAMPLE_FILE = os.environ.get("SAMPLE_FILE", "netarena_sample.json")

P = json.load(open("netarena_prompts.json", encoding="utf-8"))
TASKS = json.load(open(SAMPLE_FILE, encoding="utf-8"))


def build_prompt(question, cot):
    parts = [P["BASE_PROMPT"]]
    if cot:
        parts.append(P["COT_PROMPT"])
    parts.append(P["PROMPT_SUFFIX"].replace("{input}", question))
    return "\n".join(parts)


class _Timeout(Exception):
    pass


def _alarm(signum, frame):
    raise _Timeout()


def run_with_timeout(fn, timeout_s):
    """A hallucinated process_graph could loop forever; SIGALRM bounds it
    without forking a CUDA-holding process (fork + live CUDA context is
    unsafe if the child ever touches the GPU, so this stays in-process)."""
    old = signal.signal(signal.SIGALRM, _alarm)
    signal.alarm(timeout_s)
    try:
        return True, fn()
    except _Timeout:
        return False, "timeout after %ds" % timeout_s
    except Exception:
        return False, traceback.format_exc()
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old)


def grade_one(evaluator, task, llm_answer):
    code = extract_code_output(llm_answer)
    if not code:
        return {"correct": False, "reason": "no process_graph found in output",
                "extracted_len": 0}
    try:
        (ret, gt_ret, verifier_ok, verifier_err, gt_verifier_ok, gt_verifier_err,
         ret_graph) = evaluator.run_agent_output(task["question"], task["gt_code"], code)
        res = evaluator.ground_truth_check(
            task["question"], task["label"], ret, gt_ret, ret_graph,
            verifier_ok, verifier_err, gt_verifier_ok, gt_verifier_err, 0.0)
        # malt_env's own "Error" field is sometimes a dict (a graph-vs-graph
        # mismatch report) rather than a string, so cast up front.
        return {"correct": res.get("Result-Correctness") == "Pass",
                "reason": str(res.get("Error", "")), "extracted_len": len(code)}
    except Exception:
        return {"correct": False, "reason": "grading exception: " + traceback.format_exc()[-300:],
                "extracted_len": len(code)}


def main():
    print("loading MALT graph ...", flush=True)
    _, G = getGraphData()
    evaluator = BenchmarkEvaluator(G)
    print(f"graph loaded: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges", flush=True)

    results = []
    for repo in MODELS:
        print(f"\n=== {repo} ===", flush=True)
        entry = {"repo": repo}
        try:
            tok = AutoTokenizer.from_pretrained(repo, trust_remote_code=True)
            model = AutoModelForCausalLM.from_pretrained(
                repo, torch_dtype=torch.bfloat16, device_map="cuda:0",
                trust_remote_code=True)
            model.eval()
            cfg = model.generation_config
            cfg.do_sample = False
            for k in ("temperature", "top_p", "top_k"):
                setattr(cfg, k, None)
            if cfg.pad_token_id is None:
                cfg.pad_token_id = tok.eos_token_id

            rows = []
            t_start = time.time()
            for i, t in enumerate(TASKS):
                for cond, cot, cap in (("base", False, BASE_MAX), ("cot", True, COT_MAX)):
                    prompt = build_prompt(t["question"], cot)
                    enc = tok([tok.apply_chat_template(
                        [{"role": "user", "content": prompt}],
                        tokenize=False, add_generation_prompt=True)],
                        return_tensors="pt").to("cuda:0")
                    with torch.inference_mode():
                        o = model.generate(**enc, max_new_tokens=cap)
                    text = tok.decode(o[0][enc["input_ids"].shape[-1]:],
                                      skip_special_tokens=True)
                    ok, val = run_with_timeout(
                        lambda: grade_one(evaluator, t, text), EXEC_TIMEOUT_S)
                    grade = val if ok else {"correct": False, "reason": val, "extracted_len": 0}
                    rows.append({"task": t["id"], "label": t["label"], "cond": cond,
                                "correct": grade["correct"],
                                "reason": str(grade.get("reason", ""))[:200],
                                "extracted_len": grade.get("extracted_len", 0),
                                "raw_output": text})
                if (i + 1) % 15 == 0:
                    el = time.time() - t_start
                    print(f"  ... {i+1}/{len(TASKS)}  ({el:.0f}s elapsed)", flush=True)

            for cond in ("base", "cot"):
                sub = [r for r in rows if r["cond"] == cond]
                n_correct = sum(r["correct"] for r in sub)
                entry[cond] = {"n": len(sub), "n_correct": n_correct,
                               "accuracy": n_correct / len(sub) if sub else None,
                               "empty_extraction": sum(r["extracted_len"] == 0 for r in sub)}
                print(f"  {cond:5s}: {n_correct}/{len(sub)} correct "
                      f"({entry[cond]['accuracy']*100:.1f}%), "
                      f"{entry[cond]['empty_extraction']} empty extractions", flush=True)
            entry["rows"] = rows
            del model, tok
        except Exception:
            entry["error"] = traceback.format_exc()
            print("FAILED:\n" + entry["error"], flush=True)
        finally:
            gc.collect(); torch.cuda.empty_cache()
        results.append(entry)
        json.dump({"n_tasks": len(TASKS), "base_max": BASE_MAX, "cot_max": COT_MAX,
                   "results": results}, open(OUT, "w"), indent=1)

    print("\n=== NETARENA ACCURACY ===")
    print(f"{'model':10s} {'base acc':>9} {'cot acc':>9}")
    for e in results:
        if "base" not in e:
            continue
        print(f"{e['repo'].split('/')[-1]:22s} {e['base']['accuracy']*100:7.1f}%  "
              f"{e['cot']['accuracy']*100:7.1f}%")
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
