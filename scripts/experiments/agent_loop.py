"""What does an agentic loop cost per correct network-operations decision?

Everything this paper has measured so far is one forward pass: prompt in,
configuration out. That is not what an agent does. A real agentic deployment
plans, acts, checks its own work against something, and repairs what the check
rejects, and each of those is another generation on the same card.

This measures that. Two sources of amplification, crossed:

    reflect   the model is made to review its own answer once, always
    repair    a ground-truth-free check runs, and failures go back for a retry

giving a 2x2 against the single-shot baseline, deliberately the same shape as
the output-format 2x2 already in the paper.

The check is NOT the grader. validators.py derives its complaints from the
prompt alone -- it sees malformed and misshapen output and is blind to answers
that are well formed and simply wrong, which its own self-test demonstrates. A
loop validated against ground truth would reach 100% by construction and would
measure nothing.

Protocol is the one in reasoning_tax.py, unchanged, so the numbers compose
with the rest of the paper: per-model resident-idle baseline before and after,
foreign-process check, drift and contamination guards, trapezoidal integration
of a 10 Hz nvidia-smi stream.

Energy is measured per PASS over the whole task set rather than per task,
because a single direct generation runs 0.1-0.3 s and the paper's own third
pitfall forbids integrating a window that short. Greedy decoding is
deterministic, so replaying the set adds energy without changing any output or
any loop path; the script checks that the passes really did agree before
trusting the division.
"""
import os, gc, json, time, subprocess, statistics as st, traceback
os.environ.setdefault("HF_HOME", "/mnt/data/hf")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from power_sampler import PowerSampler
from tasks_net import build, grade
from validators import validate

# The paper's four-rung ladder, so the agentic table reads directly against
# Table 3 rather than against a subset of it.
MODELS = [
    ("Qwen/Qwen2.5-1.5B-Instruct", 1.5),
    ("Qwen/Qwen2.5-3B-Instruct",   3.0),
    ("Qwen/Qwen2.5-7B-Instruct",   7.0),
    ("Qwen/Qwen2.5-14B-Instruct",  14.0),
]

# (reflect, repair) -- the 2x2
ARMS = [("single",         False, False),
        ("repair",         False, True),
        ("reflect",        True,  False),
        ("reflect+repair", True,  True)]

# Scale is settable from the environment so that a pilot runs the experiment
# itself at reduced size rather than a separate script that could drift from
# it. PASSES=1 ONLY_MODELS=1.5B,14B ONLY_ARMS=single,repair is the pilot.
PASSES = int(os.environ.get("PASSES", 3))
MAX_REPAIRS = int(os.environ.get("MAX_REPAIRS", 3))   # k, for the repair arms
N_PER_FAMILY = int(os.environ.get("N_PER_FAMILY", 40))   # 40 x 3 = 120 tasks
SETTLE_S = int(os.environ.get("SETTLE_S", 45))  # let the floor fall back first
ANSWER_MAX = 48       # same cap as the paper's direct condition
DRIFT_W = 5.0
CONTAMINATION_W = 30.0
MY_PID = os.getpid()
DESKTOP = "gnome-remote-desktop"
OUT = os.environ.get("OUT", "agent_loop_result.json")

# Verbatim from robustness_run.py, which produced the paper's 120-task table.
# Each task keeps its assigned wording for every turn of the loop, so phrasing
# stays a property of the task rather than being reset at the repair turn.
DIRECT_TEMPLATES = [
    "\n\nDo not explain. Reply with exactly one line in the form ANSWER: <value>",
    "\n\nGive only the final answer, with no working. Format: ANSWER: <value>",
    "\n\nRespond with a single line containing just the result: ANSWER: <value>",
]
REFLECT_BODY = (
    "{prompt}\n\nA proposed answer is:\nANSWER: {answer}\n\n"
    "Check it. If it is wrong, give the corrected value; if it is right, give "
    "it again unchanged.")
REPAIR_BODY = (
    "{prompt}\n\nYour previous answer was:\nANSWER: {answer}\n\n"
    "It was rejected: {complaint}\n\nGive a corrected answer.")


def _select(env, items, key, exact):
    want = [w.strip() for w in os.environ.get(env, "").split(",") if w.strip()]
    if not want:
        return items
    if exact:                       # arm names nest: "repair" is inside
        return [it for it in items if key(it) in want]   # "reflect+repair"
    return [it for it in items if any(w in key(it) for w in want)]


MODELS = _select("ONLY_MODELS", MODELS, lambda m: m[0], exact=False)
ARMS = _select("ONLY_ARMS", ARMS, lambda a: a[0], exact=True)


def foreign_pids():
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid,process_name",
             "--format=csv,noheader"], capture_output=True, text=True,
            timeout=15).stdout
    except Exception:
        return []
    found = []
    for line in out.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 2:
            continue
        try:
            pid = int(parts[0])
        except ValueError:
            continue
        if pid == MY_PID or DESKTOP in parts[1]:
            continue
        found.append((pid, parts[1]))
    return found


def wait_for_exclusive(max_h=12):
    deadline = time.time() + max_h * 3600
    clean = 0
    while time.time() < deadline:
        fp = foreign_pids()
        if not fp:
            clean += 1
            if clean >= 3:
                return True
            time.sleep(15)
        else:
            clean = 0
            print(f"  [queue] busy: {fp}", flush=True)
            time.sleep(60)
    return False


def baseline(s, seconds, label):
    t0 = time.time(); time.sleep(seconds); t1 = time.time()
    w = s.mean_w(t0, t1)
    series = [smp[1][0] for smp in s.window(t0, t1)]
    sd = st.pstdev(series) if len(series) > 1 else 0.0
    print(f"    [{label}] {w[0]:7.2f} W (sd {sd:.2f}, n={len(series)})",
          flush=True)
    return w[0], sd


def extract(text):
    """The value after the last ANSWER: marker.

    Reuses the grader's own extraction with a sentinel ground truth, so the
    loop can never diverge from how the paper scores an answer.
    """
    _, got = grade(text, "\x00")
    return got


def generate(model, tok, prompt, max_new):
    enc = tok([tok.apply_chat_template([{"role": "user", "content": prompt}],
                                       tokenize=False,
                                       add_generation_prompt=True)],
              return_tensors="pt").to("cuda:0")
    with torch.inference_mode():
        out = model.generate(**enc, max_new_tokens=max_new)
    n_in = enc["input_ids"].shape[-1]
    gen = out[0][n_in:]
    return tok.decode(gen, skip_special_tokens=True), int(gen.shape[-1])


def run_task(model, tok, task, reflect, repair):
    """One task through the loop. Returns the trace; no energy here."""
    calls, tokens = 0, 0
    fired, repaired = 0, 0
    suffix = DIRECT_TEMPLATES[task["tpl"]]

    text, n = generate(model, tok, task["prompt"] + suffix, ANSWER_MAX)
    calls += 1; tokens += n
    answer = extract(text)

    if reflect:
        p = REFLECT_BODY.format(prompt=task["prompt"], answer=answer) + suffix
        text, n = generate(model, tok, p, ANSWER_MAX)
        calls += 1; tokens += n
        answer = extract(text)

    if repair:
        for _ in range(MAX_REPAIRS):
            ok, complaint = validate(task["family"], task["prompt"], answer)
            if ok:
                break
            fired += 1
            before = answer
            p = REPAIR_BODY.format(prompt=task["prompt"], answer=answer,
                                   complaint=complaint) + suffix
            text, n = generate(model, tok, p, ANSWER_MAX)
            calls += 1; tokens += n
            answer = extract(text)
            if answer != before:
                repaired += 1

    valid, _ = validate(task["family"], task["prompt"], answer)
    correct, _ = grade("ANSWER: " + answer, task["gt"])
    return {"task": task["id"], "family": task["family"], "calls": calls,
            "tokens": tokens, "checks_failed": fired, "answers_changed":
            repaired, "valid": bool(valid), "correct": bool(correct),
            "answer": answer}


def run_arm(model, tok, s, tasks, reflect, repair, label):
    """PASSES replays of the whole set inside one window; energy over all.

    The idle floor is taken per arm rather than once per model. A 14B pass is
    long enough for the floor to climb several watts under sustained load and
    not settle inside a short window afterwards -- the paper's own second
    pitfall -- so a single model-level baseline would charge later arms a floor
    that was measured cold. Per-arm baselines absorb the drift the way that
    pitfall's guard prescribes.
    """
    pre_w, _ = baseline(s, 12, label + " idle pre")

    torch.cuda.synchronize(); t0 = time.time()
    passes = []
    for _ in range(PASSES):
        passes.append([run_task(model, tok, t, reflect, repair) for t in tasks])
    torch.cuda.synchronize(); t1 = time.time()

    # Let the card settle before the closing floor: measured straight after a
    # long pass it reads tens of watts high and with a huge spread, which is
    # thermal lag rather than a second process.
    time.sleep(SETTLE_S)
    post_w, post_sd = baseline(s, 12, label + " idle post")

    e = s.energy_j(t0, t1)
    dur = t1 - t0
    total_j = None if e is None else e[0]
    net_j = None if e is None else e[0] - pre_w * dur

    # Greedy decoding should make every pass identical. If it did not, the
    # division by PASSES is not a division of like by like and the cell is
    # not trustworthy.
    sigs = [[(r["calls"], r["answer"]) for r in p] for p in passes]
    deterministic = all(sg == sigs[0] for sg in sigs)

    rows = passes[0]
    n = len(rows)
    return {
        "n_tasks": n, "passes": PASSES, "deterministic": deterministic,
        "pre_w": pre_w, "post_w": post_w, "post_sd": post_sd,
        "drift_w": abs(post_w - pre_w),
        "dirty": abs(post_w - pre_w) > DRIFT_W,
        "dur_s": dur, "total_j": total_j, "net_j": net_j,
        "j_per_pass": None if net_j is None else net_j / PASSES,
        "j_per_task": None if net_j is None else net_j / PASSES / n,
        "calls_per_task": sum(r["calls"] for r in rows) / n,
        "tokens_per_task": sum(r["tokens"] for r in rows) / n,
        "check_fire_rate": sum(r["checks_failed"] > 0 for r in rows) / n,
        "answer_change_rate": sum(r["answers_changed"] > 0 for r in rows) / n,
        "acc": sum(r["correct"] for r in rows) / n,
        "valid_rate": sum(r["valid"] for r in rows) / n,
        "rows": rows,
    }


def main():
    # The same 120-task set and the same round-robin template assignment that
    # robustness_run.py used for the paper's headline table, so the `single`
    # arm has to reproduce those accuracies. If it does not, something in this
    # harness has drifted and the comparison is not worth reading.
    tasks = build(n_per_family=N_PER_FAMILY)
    for i, t in enumerate(tasks):
        t["tpl"] = i % 3
    print(f"{len(tasks)} verifiable tasks, {len(DIRECT_TEMPLATES)} templates, "
          f"{PASSES} passes per arm, {len(ARMS)} arms", flush=True)

    print("waiting for an exclusive GPU ...", flush=True)
    if not wait_for_exclusive():
        print("gave up waiting")
        return

    s = PowerSampler(interval_ms=100).start()
    open_idle, _ = baseline(s, 20, "opening idle")

    results = []
    for repo, pb in MODELS:
        print(f"\n=== {repo} ({pb}B) ===", flush=True)
        entry = {"repo": repo, "params_b": pb}
        try:
            if foreign_pids():
                print("    foreign job on GPU -- waiting", flush=True)
                wait_for_exclusive()

            t_load = time.time()
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
            print(f"    loaded {time.time()-t_load:.1f} s", flush=True)

            generate(model, tok, tasks[0]["prompt"] + DIRECT_TEMPLATES[0], 8)
            torch.cuda.synchronize(); time.sleep(2)
            pre_w, _ = baseline(s, 12, "resident idle pre")

            for name, reflect, repair in ARMS:
                print(f"    -- {name}", flush=True)
                a = run_arm(model, tok, s, tasks, reflect, repair, name)
                entry[name] = a
                jc = (a["j_per_task"] / a["acc"]) if a["acc"] else float("inf")
                a["j_per_correct"] = jc
                print(f"       {a['calls_per_task']:4.2f} calls/task  "
                      f"acc {a['acc']*100:5.1f}%  "
                      f"{a['j_per_task']:8.2f} J/task  "
                      f"{jc:10.1f} J/correct  "
                      f"fire {a['check_fire_rate']*100:4.1f}%  "
                      f"drift {a['drift_w']:4.1f} W"
                      + ("" if a["deterministic"] else "  NON-DETERMINISTIC"),
                      flush=True)

            post_w, _ = baseline(s, 12, "resident idle post")
            drift = abs(post_w - pre_w)
            intruders = foreign_pids()
            entry.update(pre_w=pre_w, post_w=post_w, drift_w=drift,
                         dirty=(drift > DRIFT_W or bool(intruders)
                                or (pre_w - open_idle) > CONTAMINATION_W))
            if entry["dirty"]:
                print(f"    WARNING: cell dirty (drift {drift:.1f} W, "
                      f"foreign {intruders})", flush=True)
            del model, tok
        except Exception as exc:
            entry["error"] = f"{type(exc).__name__}: {exc}"
            print(f"    FAILED: {entry['error']}", flush=True)
            traceback.print_exc()
        finally:
            gc.collect(); torch.cuda.empty_cache(); time.sleep(3)
        results.append(entry)
        json.dump({"open_idle": open_idle, "passes": PASSES,
                   "max_repairs": MAX_REPAIRS, "results": results},
                  open(OUT, "w"), indent=1)

    print("\n=== AGENTIC AMPLIFICATION ===")
    print(f"{'model':16s} {'arm':16s} {'calls':>6} {'acc':>7} {'J/task':>9} "
          f"{'J/correct':>11} {'vs single':>10}")
    for e in results:
        base = e.get("single", {}).get("j_per_correct")
        for name, _, _ in ARMS:
            a = e.get(name)
            if not a:
                continue
            rel = (a["j_per_correct"] / base) if base else float("nan")
            print(f"{e['repo'].split('/')[-1]:16s} {name:16s} "
                  f"{a['calls_per_task']:6.2f} {a['acc']*100:6.1f}% "
                  f"{a['j_per_task']:9.2f} {a['j_per_correct']:11.1f} "
                  f"{rel:9.2f}x")
    s.stop()


if __name__ == "__main__":
    main()
