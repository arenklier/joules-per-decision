"""Does the ANSWER: extraction rule penalise chain-of-thought asymmetrically?

The grader in tasks_net.grade() takes the text after the last ANSWER: marker;
when the marker is absent it falls back to the first line of the output. A
direct answer is ~9 tokens and its first line IS the answer, so the fallback
is harmless there. A chain-of-thought answer is several hundred tokens and its
first line is reasoning, so the same fallback scores it wrong even when the
model reached the right value. If chain-of-thought omits the marker more often
than direct does, part of the measured accuracy difference is grading, not
reasoning.

This script regenerates both conditions with the same tasks, templates and
decoding as robustness_run.py, keeps the text, and reports for each condition:

  marker rate     how often ANSWER: appeared at all
  strict acc      tasks_net.grade(), i.e. what the paper reports
  last-line acc   same, but falling back to the LAST non-empty line instead of
                  the first when the marker is missing -- the reading a human
                  would give a chain-of-thought transcript
  mentions-gt     the ground truth appears anywhere in the output; a generous
                  upper bound, since a transcript can name the right value in
                  passing and still conclude wrongly

The gap between strict and last-line accuracy, restricted to marker-less
outputs, is the size of the artifact. No power measurement here: this asks a
grading question, not an energy one.
"""
import json
import os
import re
import sys
import time

os.environ.setdefault("HF_HOME", "/mnt/data/hf")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from tasks_net import build, grade

N_PER_FAMILY = int(os.environ.get("N_PER_FAMILY", 40))
DIRECT_MAX = 48
COT_MAX = int(os.environ.get("COT_MAX", 1024))
OUT = os.environ.get("OUT", "marker_audit_result.json")
MODELS = os.environ.get("MODELS", "Qwen/Qwen2.5-14B-Instruct").split(",")

# identical to robustness_run.py
DIRECT_TEMPLATES = [
    "\n\nDo not explain. Reply with exactly one line in the form ANSWER: <value>",
    "\n\nGive only the final answer, with no working. Format: ANSWER: <value>",
    "\n\nRespond with a single line containing just the result: ANSWER: <value>",
]
COT_TEMPLATES = [
    "\n\nThink step by step, showing your working. Then finish with exactly one "
    "final line in the form ANSWER: <value>",
    "\n\nWork through this carefully, one step at a time. When you are done, "
    "output the final line ANSWER: <value>",
    "\n\nReason it out in detail before committing to an answer. End your "
    "response with the line ANSWER: <value>",
]


def clean(s):
    return s.strip().strip("`*.,;:'\"").strip()


def last_line_correct(text, gt):
    """Fallback to the LAST non-empty line when no marker is present."""
    if not text:
        return False
    upper = text.upper()
    idx = upper.rfind("ANSWER:")
    if idx >= 0:                                  # marker present: same as strict
        tail = text[idx + 7:]
        got = tail.strip().splitlines()[0] if tail.strip() else ""
    else:
        lines = [l for l in text.splitlines() if l.strip()]
        got = lines[-1] if lines else ""
    return clean(got).upper() == gt.upper()


def mentions_gt(text, gt):
    if not text:
        return False
    return re.search(re.escape(gt), text, re.IGNORECASE) is not None


def encode(tok, prompt):
    return tok([tok.apply_chat_template([{"role": "user", "content": prompt}],
                                        tokenize=False, add_generation_prompt=True)],
               return_tensors="pt").to("cuda:0")


def run_condition(model, tok, tasks, templates, cap, label):
    rows = []
    t0 = time.time()
    for i, t in enumerate(tasks):
        enc = encode(tok, t["prompt"] + templates[t["tpl"]])
        with torch.inference_mode():
            out = model.generate(**enc, max_new_tokens=cap)
        gen = out[0][enc["input_ids"].shape[-1]:]
        text = tok.decode(gen, skip_special_tokens=True)
        strict, got = grade(text, t["gt"])
        rows.append({
            "task": t["id"], "family": t["family"], "tpl": t["tpl"],
            "gt": t["gt"], "new_tok": int(gen.shape[-1]),
            "marker": "ANSWER:" in text.upper(),
            "strict": bool(strict),
            "last_line": last_line_correct(text, t["gt"]),
            "mentions": mentions_gt(text, t["gt"]),
            "extracted": got[:80],
            "text": text,
        })
        if (i + 1) % 20 == 0:
            print(f"    [{label}] {i+1}/{len(tasks)}  ({time.time()-t0:.0f}s)",
                  flush=True)
    return rows


def summarize(rows, label):
    n = len(rows)
    mk = sum(r["marker"] for r in rows)
    strict = sum(r["strict"] for r in rows)
    last = sum(r["last_line"] for r in rows)
    ment = sum(r["mentions"] for r in rows)
    nomk = [r for r in rows if not r["marker"]]
    rescued = sum(1 for r in nomk if r["last_line"] and not r["strict"])
    print(f"  {label}: n={n}  marker {mk}/{n} ({mk/n*100:.1f}%)  "
          f"strict {strict/n*100:.1f}%  last-line {last/n*100:.1f}%  "
          f"mentions-gt {ment/n*100:.1f}%")
    print(f"          marker-less: {len(nomk)}  of which last-line rescues "
          f"{rescued}")
    return {"n": n, "marker": mk, "strict": strict, "last_line": last,
            "mentions": ment, "markerless": len(nomk), "rescued": rescued,
            "mean_tok": sum(r["new_tok"] for r in rows) / n}


def main():
    tasks = build(n_per_family=N_PER_FAMILY)
    for i, t in enumerate(tasks):
        t["tpl"] = i % 3
    print(f"{len(tasks)} tasks, cot cap {COT_MAX}, models: {MODELS}", flush=True)

    results = []
    for repo in MODELS:
        print(f"\n=== {repo} ===", flush=True)
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

        direct_rows = run_condition(model, tok, tasks, DIRECT_TEMPLATES,
                                    DIRECT_MAX, "direct")
        cot_rows = run_condition(model, tok, tasks, COT_TEMPLATES,
                                 COT_MAX, "cot")
        entry = {"repo": repo,
                 "direct": summarize(direct_rows, "direct"),
                 "cot": summarize(cot_rows, "cot"),
                 "direct_rows": direct_rows, "cot_rows": cot_rows}
        results.append(entry)
        json.dump({"n_tasks": len(tasks), "cot_max": COT_MAX,
                   "results": results}, open(OUT, "w"), indent=1)
        del model, tok
        import gc
        gc.collect(); torch.cuda.empty_cache()

    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    sys.exit(main())
